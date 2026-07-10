"""Standalone data loader reading LeRobot v3 datasets directly, producing
(obs_t, action_chunk, obs_t1) transitions for the JEPA + OpenPI BC train_step
in train_step_transitions.py.

WHY THIS DOESN'T REIMPLEMENT LEROBOT'S PARQUET/VIDEO READING:
LeRobot v3 stores many episodes per chunked Parquet/MP4 file and resolves
episode boundaries via metadata rather than filenames (meta/episodes/*.parquet
with data_chunk_index/data_file_index/video_chunk_index/etc, per
huggingface/lerobot's v3 docs). Reimplementing that indexing here would just
recreate a worse version of `lerobot.datasets.lerobot_dataset.LeRobotDataset`,
which already does this correctly -- including video seek-by-timestamp and
the `delta_timestamps` mechanism used below to fetch paired frames. This
module only adds the JEPA-specific transition/episode-boundary logic on top.

WHAT IS CONFIRMED vs. ASSUMED:
- Confirmed (from lerobot v3 docs + openpi source): chunked parquet/mp4
  layout; `dataset.fps`; OpenPI's Observation has `images` (dict), `state`,
  `prompt_tokens`; `Actions` is a plain [B, horizon, action_dim] array (not a
  dataclass) -- so passing a raw action-chunk array around is correct.
- ASSUMED (I could not fetch lerobot's exact v3 Python runtime API or
  openpi's transforms.py/data_loader.py): the exact attribute name lerobot
  uses at runtime for per-episode frame-index boundaries (tried a couple of
  plausible names below with a fallback -- verify against your installed
  lerobot version), and the exact call signature for applying OpenPI's
  data_transforms/model_transforms to a raw dict. That boundary is marked
  ADAPT THIS below -- fill it in from your actual transforms.py/config.py
  rather than trusting my guess.
"""

import dataclasses
import threading
import queue
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.sharding as sharding


# --------------------------------------------------------------------------- #
# 1. Episode-boundary bookkeeping
# --------------------------------------------------------------------------- #

def _last_frame_global_indices(dataset: LeRobotDataset) -> np.ndarray:
    """Global dataset indices that are the LAST frame of their episode --
    these have no valid t+1 and must be excluded from transition sampling
    (lerobot's delta_timestamps padding would otherwise silently repeat the
    same frame for t+1, giving a trivial/degenerate transition).

    Tries a couple of attribute names across lerobot versions. VERIFY this
    against your installed version (e.g. `python -c "from lerobot.datasets.lerobot_dataset
    import LeRobotDataset; d = LeRobotDataset(...); print(dir(d)); print(dir(d.meta))"`)
    if this raises or silently returns something wrong.
    """
    meta = dataset.meta
    if hasattr(meta, "episodes") and hasattr(meta.episodes, "__getitem__"):
        # v3: meta.episodes is expected to expose per-episode length/offset info
        try:
            ep_table = meta.episodes
            # Common column names seen in the v3 schema docs: 'length',
            # 'dataset_from_index', 'dataset_to_index'.
            to_idx = np.asarray(ep_table["dataset_to_index"])
            return to_idx - 1
        except Exception:
            pass
    if hasattr(dataset, "episode_data_index"):
        # v2-style API, still present in some v3 builds for back-compat.
        to_idx = np.asarray(dataset.episode_data_index["to"])
        return to_idx - 1
    raise RuntimeError(
        "Could not determine per-episode last-frame indices from this LeRobotDataset "
        "build. Inspect `dataset.meta` / `dataset.episode_data_index` in your lerobot "
        "version and adjust _last_frame_global_indices() accordingly."
    )


def _camera_keys(dataset: LeRobotDataset) -> list[str]:
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "camera_keys"):
        return list(dataset.meta.camera_keys)
    # Fallback: scan features for image-like keys.
    keys = []
    features = getattr(dataset, "features", None) or getattr(dataset.meta, "features", {})
    for k, spec in features.items():
        dtype = spec.get("dtype") if isinstance(spec, dict) else None
        if dtype in ("image", "video") or k.startswith("observation.images"):
            keys.append(k)
    if not keys:
        raise RuntimeError("Could not infer camera/image keys; set them explicitly via camera_keys=.")
    return keys


# --------------------------------------------------------------------------- #
# 2. Torch Dataset producing raw (untransformed) transitions
# --------------------------------------------------------------------------- #

class LeRobotV3TransitionDataset(Dataset):
    """Yields raw dict samples:
        {
          "images": {cam: HWC uint8 array},       # obs_t
          "state": [state_dim] float32,             # obs_t
          "action": [action_horizon, action_dim] float32,  # chunk starting at t
          "next_images": {cam: HWC uint8 array},   # obs_t1
          "next_state": [state_dim] float32,        # obs_t1
          "task": str,
        }
    Filters out the last frame of every episode (no valid t+1).
    """

    def __init__(
        self,
        repo_id_or_root: str,
        action_horizon: int,
        camera_keys: Optional[list[str]] = None,
        is_local_root: bool = True,
    ):
        base_kwargs = {"root": repo_id_or_root, "repo_id": repo_id_or_root} if is_local_root else {"repo_id": repo_id_or_root}
        # First pass: instantiate without delta_timestamps just to read fps/meta.
        probe = LeRobotDataset(**base_kwargs)
        fps = probe.fps
        step = 1.0 / fps
        self.camera_keys = camera_keys or _camera_keys(probe)

        delta_timestamps = {
            "observation.state": [0.0, step],
            "action": [i * step for i in range(action_horizon)],
        }
        for cam in self.camera_keys:
            delta_timestamps[cam] = [0.0, step]

        self.dataset = LeRobotDataset(delta_timestamps=delta_timestamps, **base_kwargs)
        last_idx = _last_frame_global_indices(self.dataset)
        all_idx = np.arange(len(self.dataset))
        self.valid_indices = np.setdiff1d(all_idx, last_idx, assume_unique=False)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        real_idx = int(self.valid_indices[i])
        item = self.dataset[real_idx]

        images_t, images_t1 = {}, {}
        for cam in self.camera_keys:
            imgs = item[cam]  # expected shape [2, C, H, W] or [2, H, W, C] depending on lerobot version
            imgs = np.asarray(imgs)
            if imgs.shape[0] != 2:
                raise RuntimeError(
                    f"Expected 2 timesteps (t, t+1) for '{cam}' from delta_timestamps, got shape {imgs.shape}. "
                    "Check that delta_timestamps was applied as expected for your lerobot version."
                )
            frame_t, frame_t1 = imgs[0], imgs[1]
            # Normalize to HWC uint8 (matches openpi's _parse_image convention
            # seen in policies/libero_policy.py: channel-first gets transposed).
            for frame, dest in ((frame_t, images_t), (frame_t1, images_t1)):
                if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[0] != frame.shape[-1]:
                    frame = np.transpose(frame, (1, 2, 0))
                if np.issubdtype(frame.dtype, np.floating):
                    frame = (255 * frame).astype(np.uint8)
                dest[cam] = frame

        state = np.asarray(item["observation.state"])
        state_t, state_t1 = state[0], state[1]

        action = np.asarray(item["action"], dtype=np.float32)  # [horizon, action_dim]

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

    ADAPT THIS FUNCTION to your actual pipeline. I don't have transforms.py /
    data_loader.py, so I can't call your real data_transforms/model_transforms
    here -- this does the minimal, best-guess version (uint8->float image
    scaling, direct dict->Observation construction, one shared tokenized
    prompt reused for both obs_t and obs_t1) instead of your dataset's actual
    normalization stats and tokenizer. Concretely: wherever your existing
    `_data_loader.create_data_loader` turns a similar dict into an
    Observation, call THAT same code path here for obs_t (so normalization/
    tokenization exactly matches training), and reuse its output's
    `prompt_tokens`/`prompt_tokens_mask` (and ideally its image
    normalization) for obs_t1 rather than duplicating logic here.
    """
    def _images_to_jax(images: dict) -> dict:
        return {
            cam: jnp.asarray(arr.astype(np.float32) / 255.0)
            for cam, arr in images.items()
        }

    obs_t = _model.Observation(
        images=_images_to_jax(raw["images"]),
        state=jnp.asarray(raw["state"]),
        prompt_tokens=None,  # ADAPT: tokenize raw["task"] with your model's tokenizer
    )
    obs_t1 = _model.Observation(
        images=_images_to_jax(raw["next_images"]),
        state=jnp.asarray(raw["next_state"]),
        prompt_tokens=obs_t.prompt_tokens,  # reuse -- same episode/task, no need to re-tokenize
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
    ):
        self.config = config
        self.data_sharding = data_sharding
        dataset = LeRobotV3TransitionDataset(
            repo_id_or_root, action_horizon, camera_keys=camera_keys, is_local_root=is_local_root
        )
        self._torch_loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
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
        while not self._stop.is_set():
            for raw in self._torch_loader:
                if self._stop.is_set():
                    return
                obs_t, action_chunk, obs_t1 = raw_batch_to_transition(raw, self.config)
                obs_t = jax.device_put(obs_t, self.data_sharding)
                action_chunk = jax.device_put(action_chunk, self.data_sharding)
                obs_t1 = jax.device_put(obs_t1, self.data_sharding)
                self._queue.put((obs_t, action_chunk, obs_t1))

    def __iter__(self) -> Iterator[tuple[_model.Observation, jnp.ndarray, _model.Observation]]:
        while True:
            yield self._queue.get()

    def close(self):
        self._stop.set()