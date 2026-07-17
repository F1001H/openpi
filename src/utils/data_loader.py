"""Standalone data loader reading LeRobot v3 datasets directly, producing
(obs_t, action_chunk, obs_t1) transitions for the JEPA + OpenPI BC train_step
in train_step_transitions.py.

REWRITE (v3): the previous raw_batch_to_transition was a hand-fabricated
Observation construction that got two things wrong (guessed field name
`prompt_tokens` instead of the real `tokenized_prompt`/`tokenized_prompt_mask`,
and applied "transforms" at the batch level when openpi's DataTransformFn
contract is explicitly per-UNBATCHED-sample). This version:
  1. Applies the REAL transform pipeline (config.data.create(...)'s
     repack_transforms -> data_transforms -> Normalize -> model_transforms)
     per-sample inside _get_transition, matching the documented contract.
  2. Uses Observation.from_dict(...) -- the actual intended construction path
     confirmed from model.py -- instead of hand-mapping dict keys to kwargs.
  3. Does NOT reimplement KoboInputs (camera remapping to the canonical
     base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb keys, image_mask
     generation for missing cameras) -- config.data.create(...) already
     returns a fully-built data_transforms Group containing a real,
     correctly-configured KoboInputs instance; this file just calls it.

STILL INFERRED, NOT CONFIRMED (no access to data_loader.py):
  - Pipeline ordering: repack -> data_transforms -> Normalize -> model_transforms.
    Justification: Normalize's norm_stats are keyed on the data-native (pre-
    padding) dimensionality, and PadStatesAndActions (a model_transform) pads
    state/actions to model_action_dim -- if Normalize ran after padding,
    `stats.mean[..., :x.shape[-1]]` would slice past what the stats actually
    have. Normalize-before-model_transforms is the only ordering consistent
    with that.
  - Prompt injection: KoboDataConfig's repack_transforms structure does NOT
    map any key to "prompt" (verified from config.py), so PromptFromLeRobotTask
    must run somewhere outside what config.data.create() returns. Since we
    already have the resolved task STRING from the LeRobot item directly
    (no need for the task_index -> tasks dict lookup PromptFromLeRobotTask
    does), we inject data["prompt"] = task_string ourselves, timed to land
    after data_transforms and before model_transforms (InjectDefaultPrompt
    no-ops if "prompt" already present; TokenizePrompt consumes it next).
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
from openpi import transforms as _transforms


# --------------------------------------------------------------------------- #
# 1. Lightweight metadata access (no full-dataset / no per-frame index)
# --------------------------------------------------------------------------- #

def _load_meta(repo_id_or_root: str, is_local_root: bool):
    kwargs = {"root": repo_id_or_root, "repo_id": repo_id_or_root} if is_local_root else {"repo_id": repo_id_or_root}
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        return LeRobotDatasetMetadata(**kwargs)
    except ImportError:
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
# 2. Real openpi transform pipeline, staged so we can inject "prompt" and
#    swap in a reduced (no-tokenization) model_transforms list for obs_t1.
# --------------------------------------------------------------------------- #

def _build_transform_stages(config: _config.TrainConfig):
    data_config = config.data.create(config.assets_dirs, config.model)
    normalize = _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)

    pre_prompt = _transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
    post_prompt_full = _transforms.compose([normalize, *data_config.model_transforms.inputs])

    # obs_t1 (the JEPA target) has no separate prompt/action-chunk target of
    # its own -- reuse obs_t's tokenized_prompt/mask instead of re-tokenizing,
    # and skip prompt-dependent transforms entirely for it.
    reduced_model_transforms = [
        t for t in data_config.model_transforms.inputs
        if not isinstance(t, (_transforms.TokenizePrompt, _transforms.TokenizeFASTInputs, _transforms.InjectDefaultPrompt))
    ]
    post_prompt_reduced = _transforms.compose([normalize, *reduced_model_transforms])

    return data_config, pre_prompt, post_prompt_full, post_prompt_reduced


# --------------------------------------------------------------------------- #
# 3. Episode-shuffled streaming dataset
# --------------------------------------------------------------------------- #

class LeRobotV3TransitionIterableDataset(IterableDataset):
    """Streams (obs_t, obs_t1) raw-but-transformed dict samples, shuffled at
    the episode level, filtering out each episode's last frame (no valid t+1
    there). Memory footprint is O(num_episodes), not O(num_frames)."""

    def __init__(
        self,
        config: _config.TrainConfig,
        repo_id_or_root: str,
        action_horizon: int,
        camera_keys: Optional[list[str]] = None,
        is_local_root: bool = False,
        shuffle_episodes: bool = True,
        seed: int = 0,
    ):
        self.config = config
        self.repo_id_or_root = repo_id_or_root
        self.action_horizon = action_horizon
        self.is_local_root = is_local_root
        self.shuffle_episodes = shuffle_episodes
        self.seed = seed

        meta = _load_meta(repo_id_or_root, is_local_root)
        self.fps = meta.fps
        self.camera_keys = camera_keys or _camera_keys_from_meta(meta)
        self.episode_ranges = _episode_frame_ranges(meta)
        self.num_episodes = len(self.episode_ranges)
        self.total_frames = int(sum(to - frm for frm, to in self.episode_ranges))
        self.approx_valid_samples = max(0, self.total_frames - self.num_episodes)

        # Transform pipeline construction is cheap (Python objects + one
        # norm_stats JSON read via config.data.create), unlike the heavy
        # LeRobotDataset build -- build it once here rather than per-worker.
        self.data_config, self._pre_prompt, self._post_prompt_full, self._post_prompt_reduced = (
            _build_transform_stages(config)
        )

    def __len__(self):
        return self.approx_valid_samples

    def _build_dataset(self) -> LeRobotDataset:
        step = 1.0 / self.fps
        delta_timestamps = {
            "observation.state": [0.0, step],
            "action": [i * step for i in range(self.action_horizon)],
        }
        for cam in self.camera_keys:
            delta_timestamps[cam] = [0.0, step]
        kwargs = {"root": self.repo_id_or_root, "repo_id": self.repo_id_or_root} if self.is_local_root else {"repo_id": self.repo_id_or_root}
        return LeRobotDataset(delta_timestamps=delta_timestamps, **kwargs)

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        if getattr(self, "_cached_dataset", None) is None:
            self._cached_dataset = self._build_dataset()
        dataset = self._cached_dataset

        rng = np.random.default_rng(self.seed + worker_id)
        episode_order = np.arange(self.num_episodes)
        if self.shuffle_episodes:
            rng.shuffle(episode_order)
        episode_order = episode_order[worker_id::num_workers]

        for ep_i in episode_order:
            frm, to = self.episode_ranges[int(ep_i)]
            if to - frm < 2:
                continue
            local_indices = np.arange(frm, to - 1)
            if self.shuffle_episodes:
                rng.shuffle(local_indices)
            for global_idx in local_indices:
                yield self._get_transition(dataset, int(global_idx))

    def _get_transition(self, dataset: LeRobotDataset, real_idx: int) -> dict:
        item = dataset[real_idx]

        # Raw per-camera frames, keyed by the ACTUAL LeRobot column names
        # (e.g. "observation.images.cam1") -- these are exactly the strings
        # KoboDataConfig's repack_transforms.structure looks up.
        imgs_t, imgs_t1 = {}, {}
        for cam in self.camera_keys:
            imgs = np.asarray(item[cam])
            if imgs.shape[0] != 2:
                raise RuntimeError(
                    f"Expected 2 timesteps (t, t+1) for '{cam}', got shape {imgs.shape}. "
                    "Check delta_timestamps handling for your lerobot version."
                )
            frame_t, frame_t1 = imgs[0], imgs[1]
            for frame, dest in ((frame_t, imgs_t), (frame_t1, imgs_t1)):
                if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[0] != frame.shape[-1]:
                    frame = np.transpose(frame, (1, 2, 0))
                if np.issubdtype(frame.dtype, np.floating):
                    # Observation.from_dict expects uint8 (it does the ->[-1,1]
                    # conversion itself); keep raw dtype through the pipeline.
                    frame = (255 * frame).astype(np.uint8)
                dest[cam] = frame

        state = np.asarray(item["observation.state"])
        state_t, state_t1 = state[0].astype(np.float32), state[1].astype(np.float32)
        action = np.asarray(item["action"], dtype=np.float32)
        task = item.get("task", "")
        if not isinstance(task, str):
            task = str(task)

        # --- obs_t: full pipeline, real action chunk, real prompt ---------- #
        raw_t = {**imgs_t, "observation.state": state_t, "action": action}
        data_t = self._pre_prompt(dict(raw_t))
        if "prompt" not in data_t:
            data_t["prompt"] = task
        data_t = self._post_prompt_full(data_t)
        self._ensure_image_mask(data_t)

        # --- obs_t1: reduced pipeline, dummy action (discarded), no tokenization --- #
        # RepackTransform's structure unconditionally looks up "action" via
        # its configured key, so a placeholder is required even though we
        # never use obs_t1's "actions" output downstream.
        raw_t1 = {**imgs_t1, "observation.state": state_t1, "action": np.zeros_like(action)}
        data_t1 = self._pre_prompt(dict(raw_t1))
        data_t1 = self._post_prompt_reduced(data_t1)
        self._ensure_image_mask(data_t1)

        return {"obs_t": data_t, "obs_t1": data_t1}

    @staticmethod
    def _ensure_image_mask(data: dict) -> None:
        """image_masks is a REQUIRED field on Observation (no default) --
        if KoboInputs doesn't produce "image_mask" for some reason, fail
        loudly rather than let Observation.from_dict KeyError somewhere
        less obvious, and offer an explicit escape hatch (all-valid mask)
        only if the caller explicitly wants that fallback behavior."""
        if "image_mask" not in data:
            raise RuntimeError(
                "Transformed sample is missing 'image_mask' -- expected KoboInputs (data_transforms) "
                "to produce it (see model.py's IMAGE_KEYS / image_mask contract). If your data_transforms "
                "genuinely don't produce this, add an explicit fallback here rather than silently "
                "assuming all-valid masks."
            )


def _stack_dicts(items: list[dict]) -> dict:
    """Recursively stack a list of (possibly nested) per-sample dicts into
    batched arrays, preserving dict structure (needed for the "image"/
    "image_mask" sub-dicts keyed by camera name)."""
    out = {}
    for k in items[0].keys():
        v0 = items[0][k]
        if isinstance(v0, dict):
            out[k] = _stack_dicts([it[k] for it in items])
        elif isinstance(v0, str):
            out[k] = [it[k] for it in items]
        else:
            out[k] = np.stack([np.asarray(it[k]) for it in items])
    return out


def _collate(batch: list[dict]) -> dict:
    return {
        "obs_t": _stack_dicts([b["obs_t"] for b in batch]),
        "obs_t1": _stack_dicts([b["obs_t1"] for b in batch]),
    }


# --------------------------------------------------------------------------- #
# 4. Batch dict -> (Observation, Actions, Observation)
# --------------------------------------------------------------------------- #

def raw_batch_to_transition(raw: dict, config: _config.TrainConfig) -> tuple[_model.Observation, jnp.ndarray, _model.Observation]:
    """Both obs_t and obs_t1 dicts are already in exactly the shape
    Observation.from_dict expects (that's what the real transform pipeline
    produces) -- no manual field-by-field construction needed anymore."""
    obs_t_dict = jax.tree.map(lambda x: x, raw["obs_t"])  # shallow copy, avoid mutating raw
    obs_t = _model.Observation.from_dict(obs_t_dict)

    obs_t1_dict = dict(raw["obs_t1"])
    # Reuse obs_t's tokenized prompt for obs_t1 (same episode/task) instead
    # of leaving it unset -- from_dict requires tokenized_prompt and
    # tokenized_prompt_mask to be provided together, and our reduced
    # pipeline deliberately produced neither for obs_t1.
    if obs_t.tokenized_prompt is not None:
        obs_t1_dict["tokenized_prompt"] = np.asarray(obs_t.tokenized_prompt)
        obs_t1_dict["tokenized_prompt_mask"] = np.asarray(obs_t.tokenized_prompt_mask)
    obs_t1 = _model.Observation.from_dict(obs_t1_dict)

    action_chunk = jnp.asarray(raw["obs_t"]["actions"])
    return obs_t, action_chunk, obs_t1


# --------------------------------------------------------------------------- #
# 5. JAX-facing infinite iterator with background prefetch + sharding
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
            config, repo_id_or_root, action_horizon, camera_keys=camera_keys,
            is_local_root=is_local_root, shuffle_episodes=True, seed=seed,
        )
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
        except Exception as e:  # noqa: BLE001
            self._queue.put(e)

    def __iter__(self) -> Iterator[tuple[_model.Observation, jnp.ndarray, _model.Observation]]:
        while True:
            item = self._queue.get()
            if isinstance(item, Exception):
                raise RuntimeError("JepaTransitionDataLoader background thread failed") from item
            yield item

    def close(self):
        self._stop.set()