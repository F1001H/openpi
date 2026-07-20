#!/usr/bin/env python3
"""One-off diagnostic: is the JEPA predictor collapsing to a near-constant
output regardless of input, or is it genuinely discriminating between
different transitions?

Metric: pairwise cosine similarity between z_pred vectors computed from N
DIFFERENT (unrelated) transitions. If the predictor collapsed to outputting
~the same vector no matter the input, off-diagonal cosine similarities will
be ~1.0. A healthy, input-sensitive predictor gives much lower similarities.
This is scale-invariant, so it isn't confounded by z_pred's overall magnitude
shrinking as training makes the scale-calibration of the fresh vision_proj/
predictor heads converge (a normal, non-collapse effect also seen in the
loss trend).

Runs the check BEFORE training (fresh init + pretrained predictor weights)
and AFTER `--num-steps` training steps, using the exact same
init_train_state/train_step machinery as scripts/train_end_to_end.py.
"""

import argparse
import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.config as _config
import openpi.training.sharding as sharding

from jepa.train_step_transitions import _get_proprio, train_step
from utils.data_loader import JepaTransitionDataLoader

from train_end_to_end import init_train_state  # noqa: E402  (scripts/ on sys.path when run as a script)

JEPA_CKPT = "/home/fabian/openpi/checkpoints/jepa_predictor/vjepa2_ac_converted.npz"


def build_zpred_fn(graphdef):
    """graphdef is fixed (same architecture before/after training) and closed
    over rather than passed as a jax.jit arg -- same pattern as
    nnx_utils.module_jit. Only `state`'s leaf VALUES differ between the
    before/after calls, so this compiles once and both calls reuse it,
    instead of eagerly dispatching the whole SigLIP+Gemma+predictor forward
    pass op-by-op (which is what made the first, unjitted version of this
    script take 15+ minutes just for the "before" check)."""

    def _fn(state, obs_t, action_chunk):
        model = nnx.merge(graphdef, state)
        action_t = model.action_proj(action_chunk[:, 0, :])
        proprio_t = model.state_proj(_get_proprio(obs_t))
        z_context = model.extract_vision_latents(obs_t)
        return model.jepa_predictor(z_context, actions=action_t[:, None, :], states=proprio_t[:, None, :])

    return jax.jit(_fn)


def _mean_offdiag_cosine_sim(vecs: np.ndarray) -> tuple[float, float]:
    """vecs: [N, F]. Returns mean pairwise cosine similarity, excluding the diagonal."""
    norm = vecs / (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    n = sim.shape[0]
    off_diag = sim[~np.eye(n, dtype=bool)]
    return float(off_diag.mean()), float(off_diag.std())


def report(label: str, model_state, jepa_loader, zpred_fn):
    obs_t, action_chunk, _obs_t1 = next(iter(jepa_loader))
    z_pred = np.asarray(zpred_fn(model_state.params, obs_t, action_chunk))  # [B, tokens, D]
    flat = z_pred.reshape(z_pred.shape[0], -1)

    raw_mean_sim, raw_std_sim = _mean_offdiag_cosine_sim(flat)
    # Raw cosine similarity is dominated by whatever shared/constant ("DC")
    # component the vectors have in common (e.g. predictor_proj's bias term)
    # -- two vectors can look ~identical by that metric even with real,
    # substantial per-input signal riding on top of a big shared offset.
    # Subtracting the batch mean before normalizing isolates the part of
    # z_pred that actually varies WITH the input, which is what "did the
    # predictor collapse to a constant" is really asking about.
    centered = flat - flat.mean(axis=0, keepdims=True)
    centered_mean_sim, centered_std_sim = _mean_offdiag_cosine_sim(centered)

    print(
        f"[{label}] z_pred shape={z_pred.shape} "
        f"mean|z_pred|={np.abs(z_pred).mean():.4f} std(z_pred over batch)={z_pred.std(axis=0).mean():.4f}\n"
        f"    raw cosine sim (dominated by shared/DC component)      ={raw_mean_sim:.4f} (std={raw_std_sim:.4f})\n"
        f"    mean-centered cosine sim (isolates input-dependent part)={centered_mean_sim:.4f} (std={centered_std_sim:.4f})"
    )
    return centered_mean_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import etils.epath as epath
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    config = _config.get_config("pi0_kobo_cube_low_mem")

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    data_config = config.data.create(config.assets_dirs, config.model)
    jepa_loader = JepaTransitionDataLoader(
        config=config,
        repo_id_or_root=str(data_config.root),
        action_horizon=config.model.action_horizon,
        data_sharding=data_sharding,
        batch_size=args.batch_size,
        num_workers=2,
        is_local_root=True,
        seed=0,
    )

    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=False, jepa_predictor_checkpoint=JEPA_CKPT,
    )
    jax.block_until_ready(train_state)

    zpred_fn = build_zpred_fn(train_state.model_def)

    print("=== BEFORE training ===")
    report("before", train_state, jepa_loader, zpred_fn)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    data_iter = iter(jepa_loader)
    batch = next(data_iter)
    for step in range(args.num_steps):
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        if step % 50 == 0 or step == args.num_steps - 1:
            print(f"step {step}: loss_jepa={float(info['loss_jepa']):.4f} loss_bc={float(info['loss_bc']):.4f}")
        batch = next(data_iter)

    print(f"=== AFTER {args.num_steps} training steps ===")
    report("after", train_state, jepa_loader, zpred_fn)

    jepa_loader.close()


if __name__ == "__main__":
    main()
