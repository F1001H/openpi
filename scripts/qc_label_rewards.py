#!/usr/bin/env python3
"""One-time reward + embedding + candidate-action labeling pass over a
LeRobot v3 dataset.

For every consecutive frame pair in every episode, using a FROZEN,
already-trained JEPA+BC checkpoint, computes:
  - JEPA prediction-error intrinsic reward (compute_intrinsic_reward, see
    src/jepa/train_step_transitions.py)
  - the pooled JEPA vision embedding of the FIRST frame (obs_t) of that pair
    (mean_pool(extract_vision_latents(obs_t))) -- this is the observation
    representation the Q-chunking critic consumes (see src/qc/critic.py).
  - `num_candidates` action chunks sampled from the frozen BC actor
    (base_model.sample_actions) at obs_t -- Phase 2's off-policy TD-target
    candidates for when this frame serves as some earlier chunk's obs_{t+h}
    (see src/qc/train_step.py's critic_loss_fn, which scores these with the
    *currently training* critic to pick the best one, rather than sampling
    live during training). Precomputing candidates here -- once, from the
    FROZEN BC actor, which never changes during critic training -- means
    critic training itself never needs the VLA model loaded at all, same
    property Phase 1 established for reward/embedding.

All three are cached to disk, one array per episode. Chunk-level aggregation
(discount-accumulating rewards over a horizon, picking embedding[t] and
embedding[t+h]) happens later, in QChunkTransitionDataset -- NOT here -- so
discount can be changed without re-running this (expensive) labeling pass.
horizon_length/action_dim, however, ARE baked into the cached candidates
(their shape) -- changing horizon_length means re-running this script.

Usage:
    uv run scripts/qc_label_rewards.py <config_name> \
        --data.root=/path/to/dataset \
        --output-path=/path/to/qc_cache.npz \
        [--checkpoint-step=N] [--reward-batch-size=32] \
        [--horizon-length=5] [--action-dim=8] [--num-candidates=8]

The checkpoint restored is config.checkpoint_dir's LATEST step by default
(pass --checkpoint-step to pick a specific one). Since this is a read-only
inference pass, it does not touch/overwrite the checkpoint directory.
"""

import argparse
import logging
import sys

import etils.epath as epath
import flax.nnx as nnx
import functools
import jax
import jax.numpy as jnp
import numpy as np
import torch.utils.data as torch_data
import openpi.transforms as _transforms
import tqdm_loggable.auto as tqdm

import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

from jepa.train_step_transitions import compute_intrinsic_reward
from utils.data_loader import (
    LeRobotV3TransitionIterableDataset,
    _collate,
    raw_batch_to_transition,
    resolve_dataset_root,
)

from train_end_to_end import init_logging, init_train_state


class _FlatTransitionDataset(torch_data.Dataset):
    """Map-style wrapper around LeRobotV3TransitionIterableDataset's
    per-sample _get_transition, flattened across all episodes into one
    stable global index -- lets a standard multi-worker DataLoader
    parallelize video decoding across processes. A naive sequential,
    single-process loop over this dataset's ~80k frames (2 timesteps x N
    cameras of decode each) was projected at ~11 hours; this is the fix.
    Each item carries (ep_i, local_idx) so results can be scattered into the
    right cache slot regardless of the (num_workers>1-induced) arrival order.
    """

    def __init__(self, ds: LeRobotV3TransitionIterableDataset):
        self.ds = ds
        self.index: list[tuple[int, int, int]] = []  # (ep_i, local_idx, global_idx)
        for ep_i, (frm, to) in enumerate(ds.episode_ranges):
            for local_idx in range(to - frm - 1):
                self.index.append((ep_i, local_idx, frm + local_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, flat_idx: int) -> dict:
        # Lazily built per-worker-process, same reasoning as
        # LeRobotV3TransitionIterableDataset.__iter__'s _cached_dataset --
        # the heavy video-decoder-backed LeRobotDataset shouldn't cross a
        # process fork.
        if getattr(self, "_cached_dataset", None) is None:
            self._cached_dataset = self.ds._build_dataset()
        ep_i, local_idx, global_idx = self.index[flat_idx]
        raw = self.ds._get_transition(self._cached_dataset, global_idx)
        raw["_ep_i"] = ep_i
        raw["_local_idx"] = local_idx
        return raw


def _collate_with_index(batch: list[dict]) -> dict:
    out = _collate(batch)
    out["_ep_i"] = np.array([b["_ep_i"] for b in batch])
    out["_local_idx"] = np.array([b["_local_idx"] for b in batch])
    return out


def _label_batch(
    config: _config.TrainConfig,
    horizon_length: int,
    action_dim: int,
    num_candidates: int,
    norm_stats: dict,
    use_quantile_norm: bool,
    state: training_utils.TrainState,
    rng: jax.Array,
    obs_t: object,
    action_t: jnp.ndarray,
    obs_t1: object,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Bundles compute_intrinsic_reward with a pooled vision-embedding
    extraction of obs_t, and candidate-action sampling, so all three come out
    of a single jit'd call sharing the same restored model -- avoids repeated
    merges/dispatches per batch. Recomputes extract_vision_latents(obs_t) a
    second time internally (compute_intrinsic_reward doesn't expose its own
    intermediate), which is wasteful compute duplication in principle, but
    this is a one-time offline script over a modest (~80k-frame) dataset --
    not worth the complexity of threading an extra return value through
    compute_intrinsic_reward for it.
    """
    reward = compute_intrinsic_reward(config, state, obs_t, action_t, obs_t1)
    model = nnx.merge(state.model_def, jax.lax.stop_gradient(state.params))
    embed_t = jnp.mean(model.extract_vision_latents(obs_t), axis=1)  # [B, tokens, D] -> [B, D]

    batch_size = action_t.shape[0]
    tiled_obs = jax.tree.map(lambda x: jnp.repeat(x, num_candidates, axis=0), obs_t)
    candidates = model.base_model.sample_actions(rng, tiled_obs, num_steps=10)
    # sample_actions' output is in NORMALIZED (z-scored) units -- must be
    # unnormalized back to native physical units before slicing to the
    # critic's native action_dim, matching Policy's real output pipeline
    # order (Unnormalize before KoboOutputs' slice). Skipping this would
    # silently feed wrong-scale actions into the critic, inconsistent with
    # action_chunk (the OTHER action input, read straight from the raw
    # dataset in native units) -- see src/qc/actor.py for the same fix
    # applied to live best-of-N sampling.
    unnormalize = _transforms.Unnormalize(norm_stats, use_quantiles=use_quantile_norm)
    candidates = unnormalize({"actions": candidates})["actions"]
    candidates = candidates[:, :horizon_length, :action_dim]  # native (unpadded) action dim
    candidates = candidates.reshape(batch_size, num_candidates, horizon_length, action_dim)

    return reward, embed_t, candidates


def label_rewards(
    config: _config.TrainConfig,
    checkpoint_step: int | None,
    batch_size: int,
    num_workers: int,
    horizon_length: int,
    action_dim: int,
    num_candidates: int,
    output_path: str,
) -> None:
    mesh = sharding.make_mesh(config.fsdp_devices)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    rng = jax.random.key(config.seed)
    rng, init_rng = jax.random.split(rng)

    # jepa_predictor_checkpoint=None: the checkpoint we're restoring already
    # has the CO-TRAINED predictor weights baked into its params (that .npz
    # is only needed once, to initialize a FRESH run from the pretrained
    # V-JEPA2-AC checkpoint) -- restore_state below overwrites everything
    # with the real trained state anyway.
    train_state_shape, _ = init_train_state(
        config, init_rng, mesh, resume=True, jepa_predictor_checkpoint=None,
    )

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True,
    )
    if not resuming:
        raise RuntimeError(
            f"No checkpoint found at {config.checkpoint_dir} to restore from -- "
            "label_rewards needs an already-trained JEPA+BC checkpoint."
        )
    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, None, step=checkpoint_step)
    logging.info(f"Restored checkpoint step={int(train_state.step)} from {config.checkpoint_dir}")

    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id_or_root, is_local_root = resolve_dataset_root(data_config)
    ds = LeRobotV3TransitionIterableDataset(
        config, repo_id_or_root, config.model.action_horizon,
        is_local_root=is_local_root, shuffle_episodes=False,
    )
    logging.info(f"{ds.num_episodes} episodes, {ds.total_frames} frames in {repo_id_or_root}")

    # Close over config/horizon_length/action_dim/num_candidates/norm_stats
    # only (matches ptrain_step's functools.partial(train_step, config)
    # convention elsewhere in this repo) -- state/rng/obs/action stay as
    # explicit traced args rather than being closed over, so their (multi-GB)
    # array values aren't baked into the compiled function as constants.
    labeling_fn = jax.jit(
        functools.partial(
            _label_batch, config, horizon_length, action_dim, num_candidates,
            data_config.norm_stats, data_config.use_quantile_norm,
        )
    )

    # Pre-allocate per-episode reward arrays (lengths known upfront from
    # episode_ranges); embed/candidate arrays allocated lazily once their
    # shapes are known from the first batch. Using a map-style DataLoader
    # with num_workers>0 parallelizes video decoding across processes -- this
    # is what makes an ~80k-frame pass tractable (a naive single-process loop
    # was ~11 hours).
    flat_ds = _FlatTransitionDataset(ds)
    loader = torch_data.DataLoader(
        flat_ds, batch_size=batch_size, num_workers=num_workers,
        collate_fn=_collate_with_index, shuffle=False,
    )

    rewards_by_episode: dict[int, np.ndarray] = {
        ep_i: np.zeros(to - frm - 1, dtype=np.float32)
        for ep_i, (frm, to) in enumerate(ds.episode_ranges) if to - frm - 1 > 0
    }
    embeds_by_episode: dict[int, np.ndarray] = {}
    candidates_by_episode: dict[int, np.ndarray] = {}

    for batched_raw in tqdm.tqdm(loader, desc="batches", total=len(flat_ds) // batch_size):
        ep_i_arr = batched_raw.pop("_ep_i")
        local_idx_arr = batched_raw.pop("_local_idx")
        obs_t, action_chunk, obs_t1 = raw_batch_to_transition(batched_raw, config)
        action_t = action_chunk[:, 0, :]

        obs_t = jax.device_put(obs_t, replicated_sharding)
        action_t = jax.device_put(action_t, replicated_sharding)
        obs_t1 = jax.device_put(obs_t1, replicated_sharding)

        rng, batch_rng = jax.random.split(rng)
        batch_rewards, batch_embeds, batch_candidates = labeling_fn(train_state, batch_rng, obs_t, action_t, obs_t1)
        batch_rewards = np.asarray(batch_rewards)
        batch_embeds = np.asarray(batch_embeds)
        batch_candidates = np.asarray(batch_candidates)

        for ep_i in np.unique(ep_i_arr):
            ep_i = int(ep_i)
            if ep_i not in embeds_by_episode:
                n = len(rewards_by_episode[ep_i])
                embeds_by_episode[ep_i] = np.zeros((n, batch_embeds.shape[-1]), dtype=np.float32)
                candidates_by_episode[ep_i] = np.zeros((n, *batch_candidates.shape[1:]), dtype=np.float32)
            sel = ep_i_arr == ep_i
            idxs = local_idx_arr[sel]
            rewards_by_episode[ep_i][idxs] = batch_rewards[sel]
            candidates_by_episode[ep_i][idxs] = batch_candidates[sel]
            embeds_by_episode[ep_i][idxs] = batch_embeds[sel]

    out_path = epath.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        **{f"episode_{ep_i}": arr for ep_i, arr in rewards_by_episode.items()},
        **{f"embed_{ep_i}": arr for ep_i, arr in embeds_by_episode.items()},
        **{f"candidates_{ep_i}": arr for ep_i, arr in candidates_by_episode.items()},
        _checkpoint_dir=str(config.checkpoint_dir),
        _checkpoint_step=int(train_state.step),
        _num_episodes=len(rewards_by_episode),
        _horizon_length=horizon_length,
        _action_dim=action_dim,
        _num_candidates=num_candidates,
    )
    logging.info(
        f"Wrote intrinsic rewards + pooled embeddings + {num_candidates} candidate actions/frame "
        f"for {len(rewards_by_episode)} episodes to {out_path}"
    )


if __name__ == "__main__":
    init_logging()
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--checkpoint-step", type=int, default=None)
    _parser.add_argument("--output-path", type=str, required=True)
    _parser.add_argument("--reward-batch-size", type=int, default=32)
    _parser.add_argument("--num-workers", type=int, default=8)
    _parser.add_argument("--horizon-length", type=int, default=5)
    _parser.add_argument("--action-dim", type=int, default=8)
    _parser.add_argument("--num-candidates", type=int, default=8)
    _args, _remaining = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    label_rewards(
        _config.cli(),
        _args.checkpoint_step,
        _args.reward_batch_size,
        _args.num_workers,
        _args.horizon_length,
        _args.action_dim,
        _args.num_candidates,
        _args.output_path,
    )
