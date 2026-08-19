#!/usr/bin/env python3
"""Serves a critic-scored best-of-N policy (src/qc/actor.py's
best_of_n_action_batch) over the same websocket protocol scripts/
serve_policy.py uses -- drop-in for any existing client (e.g.
examples/libero/main.py) that only calls policy.infer(obs)["actions"].

Offline best-of-N only: samples num_samples candidate action chunks per step
from the frozen BC/JEPA actor, scores them with a frozen, already-trained
critic (scripts/train_qc_critic.py's output), and executes the best one. No
online critic updates, no replay buffer -- that's a separate, bigger-scope
follow-up (see scripts/inference_online.py, which does that for the real
kobo robot but is real-robot-specific/untested). This script answers a
narrower question first: does critic-scored action selection actually beat
plain BC (scripts/serve_policy.py policy:checkpoint) on the same LIBERO sim
eval harness.

Pattern (model loading, critic loading, JIT-wrapped best-of-N call) copied
from scripts/inference_online.py's setup_model_and_critic/_select_action,
stripped of all ROS/real-hardware I/O. Loads the FULL OpenPIWithJEPA-wrapped
checkpoint directly via train_end_to_end.init_train_state + restore_state
(NOT policy_config.create_trained_policy, which only exposes the unwrapped
base_model -- best_of_n_action_batch needs .extract_vision_latents too).

Usage:
    uv run scripts/serve_qc_policy.py <config_name> \
        --exp-name=<exp_name> --critic-checkpoint-path=/path/to/critic/final \
        --proprio-dim=8 --action-dim=7 --horizon-length=5 --num-samples=16 \
        [--step=N] [--port=8000]

Then point examples/libero/main.py (or any openpi-client) at --port as usual.
"""

import argparse
import dataclasses
import logging
import socket
import sys

import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from flax.nnx import traversals as nnx_traversals
import orbax.checkpoint as ocp
from openpi_client import base_policy as _base_policy

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
from openpi import transforms as _transforms
from openpi.serving import websocket_policy_server

from jepa.train_step_transitions import OpenPIWithJEPA
from qc.actor import best_of_n_action_batch
from qc.checkpoint import load_critic

from train_end_to_end import init_logging, init_train_state

# JEPA predictor embed_dim -- see OpenPIWithJEPA in src/jepa/train_step_transitions.py.
EMBED_DIM = 1408


def _normalize_pure_dict_keys(tree):
    """Recursively stringifies integer dict keys. Orbax's on-disk metadata
    round-trips list-like nnx attributes (e.g. jepa_predictor.predictor_blocks,
    a plain Python list of Block submodules) back as dicts with INTEGER keys,
    while nnx.State.to_pure_dict()/flat_state() on a freshly built module use
    STRING keys for the same structure."""
    if isinstance(tree, dict):
        return {str(k): _normalize_pure_dict_keys(v) for k, v in tree.items()}
    return tree


def _load_full_jepa_model(config: _config.TrainConfig, checkpoint_dir: str):
    """Loads a flat, device-count-agnostic checkpoint produced by
    scripts/extract_full_jepa_checkpoint.py -- restores on whatever devices
    are actually present (mesh=make_mesh(1) is fine even for a checkpoint
    originally trained under N-way FSDP, since extract_full_jepa_checkpoint.py
    already re-saved it as a plain, unsharded pytree).

    Deliberately does NOT use nnx.State.replace_by_pure_dict (the pattern
    openpi.models.model.BaseModelConfig.load uses for bare BaseModel) --
    confirmed via several real restore attempts that it CANNOT round-trip
    jepa_predictor.predictor_blocks (a plain Python list, unlike anything in
    BaseModel): replace_by_pure_dict's internal try_convert_int coerces
    every numeric-looking key back to an int before comparing against
    state.flat_state(), which itself uses STRING keys for the same list --
    so even after normalizing our own params dict to strings (matching
    state's convention), flax's own function immediately undoes it and the
    lookup fails again ("key in pure_dict not available in state"). Instead,
    merge by hand via flax.nnx.traversals.flatten_mapping/unflatten_mapping
    directly on state.flat_state(), which never applies that int coercion.
    """
    rng = jax.random.key(config.seed)

    def _create(r):
        base_model = config.model.create(r)
        return OpenPIWithJEPA(base_model, config, nnx.Rngs(r))

    abstract_model = nnx.eval_shape(_create, rng)
    graphdef, state = nnx.split(abstract_model)

    params = _model.restore_params(epath.Path(checkpoint_dir) / "params", restore_type=jax.Array)
    params = _normalize_pure_dict_keys(params)
    params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)

    current_flat = state.flat_state()
    flat_params = nnx_traversals.flatten_mapping(params)
    missing = [kp for kp in flat_params if kp not in current_flat]
    if missing:
        raise RuntimeError(
            f"{len(missing)} keys from the checkpoint did not map onto the abstract OpenPIWithJEPA structure "
            f"(e.g. {missing[:5]}) -- config likely doesn't match the checkpoint's architecture."
        )
    for kp, v in flat_params.items():
        leaf = current_flat[kp]
        current_flat[kp] = leaf.replace(v) if hasattr(leaf, "replace") else v
    state.update(nnx_traversals.unflatten_mapping(current_flat))
    return nnx.merge(graphdef, state)


class QCPolicy(_base_policy.BasePolicy):
    """Critic-scored best-of-N policy, servable over the same websocket
    protocol as openpi.policies.policy.Policy."""

    def __init__(
        self,
        config: _config.TrainConfig,
        step: int | None,
        critic_checkpoint_path: str,
        num_samples: int,
        horizon_length: int,
        proprio_dim: int,
        action_dim: int,
        default_prompt: str | None,
        fsdp_devices: int = 1,
        checkpoint_dir: str | None = None,
        uncertainty_penalty: float = 0.0,
        actor_disagreement_penalty: float = 0.0,
        maximize_score: bool = False,
        selection_mode: str = "score",
        num_qs: int = 2,
    ):
        self.num_samples = num_samples
        self.horizon_length = horizon_length
        self.action_dim = action_dim
        self.uncertainty_penalty = uncertainty_penalty
        self.actor_disagreement_penalty = actor_disagreement_penalty
        self.maximize_score = maximize_score
        self.selection_mode = selection_mode

        if checkpoint_dir is not None:
            # Flat, device-count-agnostic checkpoint from scripts/
            # extract_full_jepa_checkpoint.py -- restorable on 1 GPU
            # regardless of how many devices the original training run used
            # (that script already did the N-GPU restore + flat re-save once,
            # on the cluster). This is the path for the full-finetune runs
            # (trained under 4-way FSDP) so the eval itself can run fully
            # locally, same as the plain-BC evals.
            logging.info(f"Loading flat full-JEPA checkpoint from {checkpoint_dir}")
            model = _load_full_jepa_model(config, checkpoint_dir)
            # Stop-gradient, matching the raw-TrainState branch's convention
            # below (this model is inference-only either way).
            model = nnx.merge(nnx.graphdef(model), jax.lax.stop_gradient(nnx.state(model)))
            # extract_full_jepa_checkpoint.py copies each step's own assets/
            # (norm_stats) alongside params/, same layout serve_policy.py's
            # --policy.dir checkpoints use.
            assets_root = epath.Path(checkpoint_dir) / "assets"
        else:
            # Original path: restores the RAW training TrainState directly.
            # Must match the checkpoint's OWN --fsdp-devices at training
            # time -- restoring an N-way FSDP-sharded checkpoint needs that
            # many real JAX devices actually visible, or orbax raises
            # "sharding passed to deserialization should be ... Got None".
            # Only really viable when fsdp_devices is small enough to run
            # locally (e.g. the LoRA sweep, trained on 1 GPU); for
            # full-finetune checkpoints, use --checkpoint-dir instead.
            mesh = sharding.make_mesh(fsdp_devices)
            rng = jax.random.key(config.seed)

            # jepa_predictor_checkpoint=None: the restored checkpoint already has
            # the co-trained predictor weights baked into its own params (same
            # reasoning as qc_label_rewards.py's label_rewards()).
            train_state_shape, _ = init_train_state(config, rng, mesh, resume=True, jepa_predictor_checkpoint=None)
            checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
                config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True,
            )
            if not resuming:
                raise RuntimeError(f"No checkpoint found at {config.checkpoint_dir} to restore from.")
            train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, None, step=step)
            logging.info(f"Restored BC/JEPA checkpoint step={int(train_state.step)} from {config.checkpoint_dir}")
            # Used as the `base_model` arg to best_of_n_action_batch (needs
            # .base_model.sample_actions / .extract_vision_latents) -- a
            # stop-gradiented merge, matching qc_label_rewards.py's _label_batch
            # convention.
            model = nnx.merge(train_state.model_def, jax.lax.stop_gradient(train_state.params))
            # Resolve via the CheckpointManager's own step bookkeeping, not
            # train_state.step (off-by-one from the on-disk directory name --
            # see extract_base_model_checkpoint.py's identical note) --
            # per-step assets/ (norm_stats) live under config.checkpoint_dir/
            # <step>/assets, written by save_state during training.
            resolved_step = checkpoint_manager.latest_step() if step is None else step
            assets_root = epath.Path(config.checkpoint_dir) / str(resolved_step) / "assets"

        logging.info(f"Loading critic from {critic_checkpoint_path}")
        # use_target=True: the EMA-smoothed target network is the more
        # stable choice for inference-time action scoring (train_step.py's
        # own convention for bootstrapping targets).
        critic = load_critic(
            critic_checkpoint_path, EMBED_DIM, proprio_dim, action_dim, horizon_length,
            num_qs=num_qs, use_target=True,
        )

        # Load norm_stats from the CHECKPOINT's own bundled assets/, not
        # config.assets_dirs (a repo-local path, e.g. ./assets/pi05_libero/ --
        # empty for any checkpoint whose norm stats were computed on the
        # cluster and never locally). Mirrors policy_config.create_trained_
        # policy's convention exactly; data_config here is only used for its
        # asset_id/use_quantile_norm, its own .norm_stats is NOT used.
        data_config = config.data.create(config.assets_dirs, config.model)
        self.use_quantile_norm = data_config.use_quantile_norm
        if data_config.asset_id is None:
            raise ValueError(f"{config.name}'s DataConfig has no asset_id -- can't resolve norm_stats.")
        self.norm_stats = _checkpoints.load_norm_stats(assets_root, data_config.asset_id)
        if self.norm_stats is None:
            raise RuntimeError(f"No norm_stats found at {assets_root / data_config.asset_id} -- actions would go through un-normalized.")
        normalize = _transforms.Normalize(self.norm_stats, use_quantiles=self.use_quantile_norm)

        # Real input pipeline, minus repack_transforms: examples/libero/
        # main.py's element dict is already "observation/image"-keyed, same
        # convention policy_config.create_trained_policy's default (empty)
        # repack_transforms relies on.
        self.input_transform = _transforms.compose(
            [
                _transforms.InjectDefaultPrompt(default_prompt),
                *data_config.data_transforms.inputs,
                normalize,
                *data_config.model_transforms.inputs,
            ]
        )

        # JIT-wrap the per-step action-selection call -- found NECESSARY, not
        # just a speedup, in scripts/inference_online.py's live dry run:
        # calling best_of_n_action_batch as a bare Python function runs every
        # internal op eagerly (no jit boundary to reuse across calls), and
        # GPU memory usage grew iteration over iteration until a
        # RESOURCE_EXHAUSTED crash.
        self._model_graphdef, self._model_state = nnx.split(model)
        self._critic_graphdef, self._critic_state = nnx.split(critic)

        def _select_action(model_state, critic_state, rng, obs, proprio):
            m = nnx.merge(self._model_graphdef, model_state)
            c = nnx.merge(self._critic_graphdef, critic_state)
            return best_of_n_action_batch(
                rng, m, c, obs, proprio,
                self.num_samples, self.horizon_length, self.action_dim,
                self.norm_stats, use_quantile_norm=self.use_quantile_norm,
                uncertainty_penalty=self.uncertainty_penalty,
                actor_disagreement_penalty=self.actor_disagreement_penalty,
                maximize_score=self.maximize_score,
                selection_mode=self.selection_mode,
            )

        self._select_action_jit = jax.jit(_select_action)
        self._rng = jax.random.key(config.seed + 1)

    def infer(self, obs: dict) -> dict:
        inputs = jax.tree.map(lambda x: x, obs)
        # proprio is RAW native units (no Normalize), matching
        # QChunkTransitionDataset's own proprio_t convention (src/utils/
        # data_loader.py), which the critic was trained against -- must be
        # read BEFORE input_transform (which normalizes "state" in place).
        proprio_native = jnp.asarray(np.asarray(inputs["observation/state"], dtype=np.float32))[np.newaxis, ...]

        data = self.input_transform(dict(inputs))
        data = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], data)
        observation = _model.Observation.from_dict(data)

        self._rng, action_rng = jax.random.split(self._rng)
        best_actions = self._select_action_jit(
            self._model_state, self._critic_state, action_rng, observation, proprio_native,
        )
        actions = np.asarray(best_actions[0])  # [horizon_length, action_dim], native units
        return {"actions": actions}


def main(
    config: _config.TrainConfig,
    step: int | None,
    critic_checkpoint_path: str,
    num_samples: int,
    horizon_length: int,
    proprio_dim: int,
    action_dim: int,
    default_prompt: str | None,
    port: int,
    fsdp_devices: int = 1,
    checkpoint_dir: str | None = None,
    uncertainty_penalty: float = 0.0,
    actor_disagreement_penalty: float = 0.0,
    maximize_score: bool = False,
    selection_mode: str = "score",
    num_qs: int = 2,
) -> None:
    init_logging()
    policy = QCPolicy(
        config, step, critic_checkpoint_path, num_samples, horizon_length, proprio_dim, action_dim, default_prompt,
        fsdp_devices=fsdp_devices, checkpoint_dir=checkpoint_dir, uncertainty_penalty=uncertainty_penalty,
        actor_disagreement_penalty=actor_disagreement_penalty, maximize_score=maximize_score,
        selection_mode=selection_mode, num_qs=num_qs,
    )
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Creating server (host: {hostname}, ip: {local_ip})")
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=port, metadata={})
    server.serve_forever()


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--step", type=int, default=None, help="Checkpoint step to restore (default: latest).")
    _parser.add_argument("--critic-checkpoint-path", type=str, required=True)
    _parser.add_argument("--num-samples", type=int, default=16)
    _parser.add_argument("--horizon-length", type=int, default=5)
    _parser.add_argument("--proprio-dim", type=int, default=8)
    _parser.add_argument("--action-dim", type=int, default=7)
    _parser.add_argument("--default-prompt", type=str, default=None)
    _parser.add_argument("--port", type=int, default=8000)
    _parser.add_argument("--fsdp-devices", type=int, default=1)
    _parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Flat checkpoint dir from scripts/extract_full_jepa_checkpoint.py -- restorable on 1 GPU regardless "
        "of the original training run's --fsdp-devices. If set, --step/--fsdp-devices/--exp-name are ignored for "
        "model restore (config is still used for data_config/norm_stats).",
    )
    _parser.add_argument(
        "--uncertainty-penalty", type=float, default=0.0,
        help="Weight on cross-Q-head disagreement added to the (minimized) selection score -- see qc/actor.py's "
        "best_of_n_action_batch. 0.0 (default) is plain argmin(q); positive values additionally bias toward "
        "candidates the critic's num_qs heads agree on.",
    )
    _parser.add_argument(
        "--actor-disagreement-penalty", type=float, default=0.0,
        help="Weight on actor-side sample disagreement (RMS distance of each candidate from the num_samples "
        "candidates' own mean at that observation) added to the selection score. Distinct from --uncertainty-"
        "penalty (critic Q-head disagreement) -- this instead penalizes candidates that are outliers relative "
        "to what the BC actor itself mostly proposed here.",
    )
    _parser.add_argument(
        "--maximize-score", action="store_true", default=False,
        help="Select argmax instead of the default argmin over the (possibly uncertainty-penalized) predicted "
        "JEPA prediction-error score -- restores the original (pre-fix) best-of-N direction. Off by default; "
        "flip on only if an argmin-vs-argmax A/B on real eval data shows argmax actually wins.",
    )
    _parser.add_argument(
        "--selection-mode", type=str, default="score", choices=["score", "majority_vote"],
        help="'score' (default): aggregate critic heads (q_agg) into one score, optionally penalized by "
        "--uncertainty-penalty/--actor-disagreement-penalty, then argmin/argmax. 'majority_vote': each critic "
        "head votes independently for its own best candidate (per --maximize-score's direction), plurality wins "
        "-- ignores q_agg/uncertainty_penalty/actor_disagreement_penalty, which are alternative ways of "
        "combining heads that voting replaces. Only meaningful with num_qs > 2 (see train_qc_critic.py --num-qs).",
    )
    _parser.add_argument(
        "--num-qs", type=int, default=2,
        help="Must match the critic checkpoint's own --num-qs at training time (train_qc_critic.py) -- needed to "
        "build the matching abstract structure for restore.",
    )
    _args, _remaining = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    main(
        _config.cli(),
        _args.step,
        _args.critic_checkpoint_path,
        _args.num_samples,
        _args.horizon_length,
        _args.proprio_dim,
        _args.action_dim,
        _args.default_prompt,
        _args.port,
        fsdp_devices=_args.fsdp_devices,
        checkpoint_dir=_args.checkpoint_dir,
        uncertainty_penalty=_args.uncertainty_penalty,
        actor_disagreement_penalty=_args.actor_disagreement_penalty,
        maximize_score=_args.maximize_score,
        selection_mode=_args.selection_mode,
        num_qs=_args.num_qs,
    )
