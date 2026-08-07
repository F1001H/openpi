#!/usr/bin/env python3
"""Offline Q-chunking critic training -- Phase 1 (critic only, no actor, no
online rollout). See the approved plan at
.claude/plans/velvet-whistling-gosling.md and src/qc/train_step.py's module
docstring for the full design rationale (embedding/reward caching, SARSA-style
target, why there's no actor network here yet).

Prerequisite: run scripts/qc_label_rewards.py first to produce the
qc-cache-path this script consumes.

Usage:
    uv run scripts/train_qc_critic.py <config_name> \
        --data.root=/path/to/dataset \
        --qc-cache-path=/path/to/qc_cache.npz \
        [--horizon-length=5] [--discount=0.99] [--tau=0.005] [--lr=3e-4] \
        [--proprio-dim=8] [--action-dim=8] \
        [--batch-size=256] [--num-train-steps=10000] [--log-interval=100] \
        [--checkpoint-dir=...] [--no-wandb-enabled]

--proprio-dim/--action-dim must match the dataset's NATIVE (unpadded)
observation.state/action dims (see /meta/info.json's feature shapes, NOT
config.model.action_dim -- Pi0's padded space) AND the --action-dim that
scripts/qc_label_rewards.py was run with (it bakes candidate-action-chunk
shape into the cache). Defaults (8/8) match kobo; Libero is 8/7.
"""

import argparse
import logging
import sys

import etils.epath as epath
import jax
import orbax.checkpoint as ocp
import tqdm_loggable.auto as tqdm
import wandb

import openpi.training.config as _config
import openpi.training.sharding as sharding

from qc.train_step import init_qc_train_state, train_step
from utils.data_loader import QChunkDataLoader, resolve_dataset_root

from train_end_to_end import init_logging

# JEPA predictor embed_dim -- see OpenPIWithJEPA in src/jepa/train_step_transitions.py.
# Must match whatever the checkpoint used by scripts/qc_label_rewards.py was
# trained with (all current configs use the ViT-g V-JEPA2-AC default, 1408).
EMBED_DIM = 1408


def main(
    config: _config.TrainConfig,
    qc_cache_path: str,
    horizon_length: int,
    discount: float,
    tau: float,
    lr: float,
    num_qs: int,
    proprio_dim: int,
    action_dim: int,
    batch_size: int,
    num_train_steps: int,
    log_interval: int,
    checkpoint_dir: str | None,
    wandb_enabled: bool,
):
    init_logging()

    if wandb_enabled:
        wandb.init(
            project="qc-critic",
            config=dict(
                horizon_length=horizon_length, discount=discount, tau=tau, lr=lr,
                num_qs=num_qs, batch_size=batch_size,
            ),
        )
    else:
        wandb.init(mode="disabled")

    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id_or_root, is_local_root = resolve_dataset_root(data_config)

    loader = QChunkDataLoader(
        config, repo_id_or_root, horizon_length, qc_cache_path,
        data_sharding=data_sharding, batch_size=batch_size, discount=discount,
        num_workers=2, is_local_root=is_local_root,
    )
    data_iter = iter(loader)

    rng = jax.random.key(config.seed)
    state = init_qc_train_state(
        rng, EMBED_DIM, proprio_dim, action_dim, horizon_length,
        lr=lr, tau=tau, discount=discount, num_qs=num_qs,
    )

    pbar = tqdm.tqdm(range(num_train_steps), dynamic_ncols=True)
    try:
        for step in pbar:
            batch = next(data_iter)
            state, info = train_step(state, batch)
            if step % log_interval == 0:
                info_str = ", ".join(f"{k}={float(v):.4f}" for k, v in info.items())
                pbar.write(f"Step {step}: {info_str}")
                wandb.log({k: float(v) for k, v in info.items()}, step=step)
    finally:
        loader.close()

    if checkpoint_dir:
        out = epath.Path(checkpoint_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Save nnx.State directly (not to_pure_dict()) -- orbax handles it as
        # a regular pytree, and restoring against an abstract template built
        # the same way (see qc/checkpoint.py's load_critic) round-trips
        # cleanly. Verified locally: save -> restore -> nnx.merge -> forward
        # pass all work with this pattern.
        ocp.PyTreeCheckpointer().save(
            str(out / "final"),
            {"params": state.params, "target_params": state.target_params, "step": state.step},
        )
        logging.info(f"Saved critic checkpoint to {out / 'final'}")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--qc-cache-path", type=str, required=True)
    _parser.add_argument("--horizon-length", type=int, default=5)
    _parser.add_argument("--discount", type=float, default=0.99)
    _parser.add_argument("--tau", type=float, default=0.005)
    _parser.add_argument("--lr", type=float, default=3e-4)
    _parser.add_argument("--num-qs", type=int, default=2)
    _parser.add_argument("--proprio-dim", type=int, default=8)
    _parser.add_argument("--action-dim", type=int, default=8)
    _parser.add_argument("--batch-size", type=int, default=256)
    _parser.add_argument("--num-train-steps", type=int, default=10_000)
    _parser.add_argument("--log-interval", type=int, default=100)
    _parser.add_argument("--checkpoint-dir", type=str, default=None)
    _parser.add_argument("--no-wandb-enabled", dest="wandb_enabled", action="store_false", default=True)
    _args, _remaining = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    main(
        _config.cli(),
        _args.qc_cache_path,
        _args.horizon_length,
        _args.discount,
        _args.tau,
        _args.lr,
        _args.num_qs,
        _args.proprio_dim,
        _args.action_dim,
        _args.batch_size,
        _args.num_train_steps,
        _args.log_interval,
        _args.checkpoint_dir,
        _args.wandb_enabled,
    )
