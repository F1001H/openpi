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

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from openpi_client import base_policy as _base_policy

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
from openpi import transforms as _transforms
from openpi.serving import websocket_policy_server

from qc.actor import best_of_n_action_batch
from qc.checkpoint import load_critic

from train_end_to_end import init_logging, init_train_state

# JEPA predictor embed_dim -- see OpenPIWithJEPA in src/jepa/train_step_transitions.py.
EMBED_DIM = 1408


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
    ):
        self.num_samples = num_samples
        self.horizon_length = horizon_length
        self.action_dim = action_dim

        mesh = sharding.make_mesh(1)
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

        logging.info(f"Loading critic from {critic_checkpoint_path}")
        # use_target=True: the EMA-smoothed target network is the more
        # stable choice for inference-time action scoring (train_step.py's
        # own convention for bootstrapping targets).
        critic = load_critic(
            critic_checkpoint_path, EMBED_DIM, proprio_dim, action_dim, horizon_length, use_target=True,
        )

        data_config = config.data.create(config.assets_dirs, config.model)
        self.norm_stats = data_config.norm_stats
        self.use_quantile_norm = data_config.use_quantile_norm
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
) -> None:
    init_logging()
    policy = QCPolicy(
        config, step, critic_checkpoint_path, num_samples, horizon_length, proprio_dim, action_dim, default_prompt,
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
    )
