"""Standalone data loader reading LeRobot v3 datasets directly, producing
(obs_t, action_chunk, obs_t1) transitions for the JEPA + OpenPI BC train_step
in train_step_transitions.py.

MEMORY DESIGN (v2 of this file): this is an IterableDataset that shuffles at
the EPISODE level and streams frames within each episode lazily, rather than
a map-style Dataset backed by a global per-frame index array. A prior version
did `np.arange(len(dataset))` + `np.setdiff1d(...)` to filter out
last-frame-of-episode indices -- for a dataset with tens/hundreds of millions
of frames that's real, avoidable memory, since it's proportional to frame
count. Episode metadata (a handful of ints per episode) is orders of
magnitude smaller than frame-level metadata, so indexing by episode instead
of by frame is what actually fixes the scaling, not just a smaller constant.

The underlying `LeRobotDataset` itself is not the source of the RAM concern:
LeRobot v3's tabular data is Arrow/Parquet, accessed memory-mapped, and video
frames are decoded on-demand by seeking to a timestamp rather than decoding
the whole file upfront (that's the documented point of the v3 format). This
file's job is just to not undo that by building its own O(num_frames)
in-memory structure on top.

WHAT IS CONFIRMED vs. ASSUMED -- same caveats as before, plus:
- ASSUMED: a lightweight metadata-only class (`LeRobotDatasetMetadata` or
  similar) exists in your lerobot version for reading fps/camera_keys/episode
  ranges without constructing a full LeRobotDataset. I couldn't confirm the
  exact class name for v3 from what I could fetch. There's a fallback that
  constructs a full LeRobotDataset just to read `.meta` if the lightweight
  class isn't importable -- correct, but heavier than necessary. If you find
  the exact lightweight class in your installed version, swap it in.
- If RAM still looks wrong after this change, the next thing to check is
  whether your specific installed lerobot version's LeRobotDataset.__init__
  eagerly materializes something it document says is lazy (pre-release v3
  builds have had bugs like this) -- worth a `tracemalloc`/`memory_profiler`
  pass rather than guessing further from here.
"""

import queue
import threading
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import IterableDataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

import openpi.models.model as _model
import openpi.training.config as _config


# --------------------------------------------------------------------------- #
# 1. Lightweight metadata access (no full-dataset / no per-frame index)
# --------------------------------------------------------------------------- #

def _load_meta(repo_id_or_root: str, is_local_root: bool):
    kwargs = {"root": repo_id_or_root} if is_local_root else {"repo_id": repo_id_or_root}
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        return LeRobotDatasetMetadata(**kwargs)
    except ImportError:
        # Heavier fallback: builds a full dataset just to read .meta.
        return LeRobotDataset(**kwargs).meta


def _camera_keys_from_meta(meta) -> list[str]:
    if hasattr(meta, "camera_keys"):
        return list(meta.camera_keys)
    keys = []
    features = getattr(meta, "features", {})
    for k, spec in features.items():
        dtype = spec.get("dtype") if isinstance(spec, dict) else None
        if dtype in ("image", "video") or k.startswith("observation.images"):
            keys.append(k)
    if not keys:
        raise RuntimeError("Could not infer camera/image keys; pass camera_keys= explicitly.")
    return keys


def _episode_frame_ranges(meta) -> list[tuple[int, int]]:
    """Per-episode (from, to) EXCLUSIVE global frame-index ranges. Size is
    O(num_episodes), not O(num_frames) -- this is the whole point.
    VERIFY column/attribute names against your installed lerobot version if
    this raises (see module docstring)."""
    if hasattr(meta, "episodes"):
        try:
            ep_table = meta.episodes
            froms = np.asarray(ep_table["dataset_from_index"])
            tos = np.asarray(ep_table["dataset_to_index"])
            return list(zip(froms.tolist(), tos.tolist()))
        except Exception:
            pass
    if hasattr(meta, "episode_data_index"):
        froms = np.asarray(meta.episode_data_index["from"])
        tos = np.asarray(meta.episode_data_index["to"])
        return list(zip(froms.tolist(), tos.tolist()))
    raise RuntimeError(
        "Could not determine per-episode frame ranges from this metadata object. "
        "Inspect `meta` (`print(dir(meta))`) in your lerobot version and adjust "
        "_episode_frame_ranges() accordingly."
    )


# --------------------------------------------------------------------------- #
# 2. Episode-shuffled streaming dataset
# --------------------------------------------------------------------------- #

class LeRobotV3TransitionIterableDataset(IterableDataset):
    """Streams (obs_t, action_chunk, obs_t1) raw dict samples, shuffled at
    the episode level, filtering out each episode's last frame (no valid t+1
    there). Memory footprint is O(num_episodes), not O(num_frames)."""

    def __init__(
        self,
        repo_id_or_root: str,
        action_horizon: int,
        camera_keys: Optional[list[str]] = None,
        is_local_root: bool = False,
        shuffle_episodes: bool = True,
        seed: int = 0,
    ):
        self.repo_id_or_root = repo_id_or_root
        self.action_horizon = action_horizon
        self.is_local_root = is_local_root
        self.shuffle_episodes = shuffle_episodes
        self.seed = seed

        meta = _load_meta(repo_id_or_root, is_local_root)
        self.fps = meta.fps
        self.camera_keys = camera_keys or _camera_keys_from_meta(meta)
        self.episode_ranges = _episode_frame_ranges(meta)  # O(num_episodes)
        self.num_episodes = len(self.episode_ranges)
        # cheap: sum over a num_episodes-length list, not a per-frame array
        self.total_frames = int(sum(to - frm for frm, to in self.episode_ranges))
        self.approx_valid_samples = max(0, self.total_frames - self.num_episodes)

    def __len__(self):
        # Approximate (excludes empty/1-frame episodes edge cases) -- for
        # logging/progress-bar purposes only, not used for indexing.
        return self.approx_valid_samples

    def _build_dataset(self) -> LeRobotDataset:
        step = 1.0 / self.fps
        delta_timestamps = {
            "observation.state": [0.0, step],
            "action": [i * step for i in range(self.action_horizon)],
        }
        for cam in self.camera_keys:
            delta_timestamps[cam] = [0.0, step]
        kwargs = {"root": self.repo_id_or_root} if self.is_local_root else {"repo_id": self.repo_id_or_root}
        return LeRobotDataset(delta_timestamps=delta_timestamps, **kwargs)

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # __iter__ is called once per EPOCH (PyTorch re-invokes iter(dataset)
        # at every epoch boundary even with persistent_workers=True), so
        # rebuilding the full LeRobotDataset here every time would silently
        # undo the point of the lightweight-metadata split above -- the heavy
        # construction (not just the metadata probe) would repeat every
        # epoch. Cache it on self instead: persistent worker processes keep
        # their copy of self alive across epochs, so this is a true
        # once-per-worker-process cost, not once-per-epoch.
        if getattr(self, "_cached_dataset", None) is None:
            self._cached_dataset = self._build_dataset()
        dataset = self._cached_dataset

        rng = np.random.default_rng(self.seed + worker_id)
        episode_order = np.arange(self.num_episodes)
        if self.shuffle_episodes:
            rng.shuffle(episode_order)
        # shard episodes across workers so they don't duplicate work
        episode_order = episode_order[worker_id::num_workers]

        for ep_i in episode_order:
            frm, to = self.episode_ranges[int(ep_i)]
            if to - frm < 2:
                continue  # no valid t -> t+1 pair possible in a 0/1-frame episode
            local_indices = np.arange(frm, to - 1)  # exclude last frame
            if self.shuffle_episodes:
                rng.shuffle(local_indices)
            for global_idx in local_indices:
                yield self._get_transition(dataset, int(global_idx))

    def _get_transition(self, dataset: LeRobotDataset, real_idx: int) -> dict:
        item = dataset[real_idx]

        images_t, images_t1 = {}, {}
        for cam in self.camera_keys:
            imgs = np.asarray(item[cam])  # expected [2, ...] from delta_timestamps
            if imgs.shape[0] != 2:
                raise RuntimeError(
                    f"Expected 2 timesteps (t, t+1) for '{cam}', got shape {imgs.shape}. "
                    "Check delta_timestamps handling for your lerobot version."
                )
            frame_t, frame_t1 = imgs[0], imgs[1]
            for frame, dest in ((frame_t, images_t), (frame_t1, images_t1)):
                if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[0] != frame.shape[-1]:
                    frame = np.transpose(frame, (1, 2, 0))
                if np.issubdtype(frame.dtype, np.floating):
                    frame = (255 * frame).astype(np.uint8)
                dest[cam] = frame

        state = np.asarray(item["observation.state"])
        state_t, state_t1 = state[0], state[1]
        action = np.asarray(item["action"], dtype=np.float32)
        task = item.get("task", "")

        return {
            "images": images_t,
            "state": state_t.astype(np.float32),
            "action": action,
            "next_images": images_t1,
            "next_state": state_t1.astype(np.float32),
            "task": task,
        }


def _collate(batch: list[dict]) -> dict:
    out = {}
    cams = batch[0]["images"].keys()
    out["images"] = {cam: np.stack([b["images"][cam] for b in batch]) for cam in cams}
    out["next_images"] = {cam: np.stack([b["next_images"][cam] for b in batch]) for cam in cams}
    out["state"] = np.stack([b["state"] for b in batch])
    out["next_state"] = np.stack([b["next_state"] for b in batch])
    out["action"] = np.stack([b["action"] for b in batch])
    out["task"] = [b["task"] for b in batch]
    return out


# --------------------------------------------------------------------------- #
# 3. ADAPT THIS: raw dict -> (Observation, Actions, Observation)
# --------------------------------------------------------------------------- #

def raw_batch_to_transition(raw: dict, config: _config.TrainConfig) -> tuple[_model.Observation, jnp.ndarray, _model.Observation]:
    """Convert a raw collated batch into (obs_t, action_chunk, obs_t1).

    ADAPT THIS FUNCTION to your actual pipeline -- see prior notes. This does
    the minimal best-guess version instead of your real data_transforms/
    model_transforms (normalization stats, tokenizer). Wherever your existing
    `_data_loader.create_data_loader` builds an Observation from a similar
    dict, call that same path here for obs_t, and reuse its prompt_tokens for
    obs_t1 rather than duplicating logic.
    """
    def _images_to_jax(images: dict) -> dict:
        return {cam: jnp.asarray(arr.astype(np.float32) / 255.0) for cam, arr in images.items()}

    obs_t = _model.Observation(
        images=_images_to_jax(raw["images"]),
        state=jnp.asarray(raw["state"]),
        prompt_tokens=None,  # ADAPT: tokenize raw["task"] with your model's tokenizer
    )
    obs_t1 = _model.Observation(
        images=_images_to_jax(raw["next_images"]),
        state=jnp.asarray(raw["next_state"]),
        prompt_tokens=obs_t.prompt_tokens,  # reuse -- same episode/task
    )
    action_chunk = jnp.asarray(raw["action"])
    return obs_t, action_chunk, obs_t1


# --------------------------------------------------------------------------- #
# 4. JAX-facing infinite iterator with background prefetch + sharding
# --------------------------------------------------------------------------- #

class JepaTransitionDataLoader:
    def __init__(
        self,
        config: _config.TrainConfig,
        repo_id_or_root: str,
        action_horizon: int,
        data_sharding: jax.sharding.NamedSharding,
        batch_size: int,
        num_workers: int = 4,
        camera_keys: Optional[list[str]] = None,
        is_local_root: bool = False,
        prefetch: int = 4,
        seed: int = 0,
    ):
        self.config = config
        self.data_sharding = data_sharding
        dataset = LeRobotV3TransitionIterableDataset(
            repo_id_or_root, action_horizon, camera_keys=camera_keys,
            is_local_root=is_local_root, shuffle_episodes=True, seed=seed,
        )
        # NOTE: no shuffle= kwarg here -- IterableDataset shuffles itself
        # (at the episode level, inside __iter__); torch.utils.data.DataLoader
        # disallows shuffle=True combined with an IterableDataset.
        self._torch_loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=_collate,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
        self._queue: "queue.Queue" = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            while not self._stop.is_set():
                for raw in self._torch_loader:
                    if self._stop.is_set():
                        return
                    obs_t, action_chunk, obs_t1 = raw_batch_to_transition(raw, self.config)
                    obs_t = jax.device_put(obs_t, self.data_sharding)
                    action_chunk = jax.device_put(action_chunk, self.data_sharding)
                    obs_t1 = jax.device_put(obs_t1, self.data_sharding)
                    self._queue.put((obs_t, action_chunk, obs_t1))
        except Exception as e:  # noqa: BLE001 -- deliberately broad: forward *any* failure
            # Without this, an exception here (bad frame, decode error, OOM,
            # etc.) just kills this daemon thread silently -- the main loop's
            # queue.get() then blocks forever with no error, no traceback
            # anywhere obvious, and no indication training has actually
            # stopped making progress. Put it on the queue instead so the
            # consumer sees the failure the next time it asks for a batch.
            self._queue.put(e)

    def __iter__(self) -> Iterator[tuple[_model.Observation, jnp.ndarray, _model.Observation]]:
        while True:
            item = self._queue.get()
            if isinstance(item, Exception):
                raise RuntimeError("JepaTransitionDataLoader background thread failed") from item
            yield item

    def close(self):
        self._stop.set()