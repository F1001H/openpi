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

# PyTorch's default ("file_descriptor") multiprocessing sharing strategy
# passes worker->main-process tensors (image batches, here) through
# /dev/shm -- on cluster nodes where that's allocated small (common under
# SLURM/containers), num_workers>0 DataLoaders crash with "Unexpected bus
# error ... insufficient shared memory (shm)" (SIGBUS) once enough data is
# in flight. "file_system" instead uses temp files for the same handoff,
# sidestepping the /dev/shm size entirely -- the standard fix for this exact
# symptom. Must be set before any DataLoader with workers is constructed,
# hence at import time here (both JepaTransitionDataLoader and
# QChunkDataLoader below use TorchDataLoader with num_workers>0 by default).
torch.multiprocessing.set_sharing_strategy("file_system")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

import openpi.models.model as _model
import openpi.training.config as _config
from openpi import transforms as _transforms


def resolve_dataset_root(data_config: _config.DataConfig) -> tuple[str, bool]:
    """Resolves a created DataConfig to (repo_id_or_root, is_local_root) the
    way LeRobotDataset/LeRobotDatasetMetadata expect -- handling both
    local-root configs (e.g. KoboDataConfig, root set via --data.root=... on
    the CLI) and Hub repo_id configs (e.g. LeRobotLiberoDataConfig,
    repo_id="physical-intelligence/libero"). Prefers root over repo_id when
    both are set (matches KoboDataConfig-style local configs, which also set
    a "local/..." placeholder repo_id).

    NOTE: DataConfigFactory.create_base_config() always sets
    root=self.root, unconditionally overwriting whatever root was set on
    base_config -- for a KoboDataConfig-style config, the only way to
    actually get a local root through right now is --data.root=/path on the
    CLI (a base_config=DataConfig(root=...) value is silently discarded)."""
    if data_config.root is not None:
        return str(data_config.root), True
    if data_config.repo_id is not None:
        if data_config.repo_id.startswith("local/"):
            raise ValueError(
                f"config.data.repo_id ('{data_config.repo_id}') looks like a local-only placeholder, "
                f"not a real Hub repo id, and config.data.root is None. Pass --data.root=/path/to/dataset "
                f"on the CLI."
            )
        return data_config.repo_id, False
    raise ValueError("config.data resolved to neither a root path nor a repo_id.")


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

def _infer_raw_state_action_keys(data_config: _config.DataConfig) -> tuple[str, str]:
    """Reads the RAW (pre-repack) dataset column names for state/action off
    data_config.repack_transforms' RepackTransform.structure -- the single
    source of truth every DataConfigFactory (KoboDataConfig,
    LeRobotLiberoDataConfig, ...) already defines this mapping in (structure
    maps canonical keys like "observation/state"/"actions" to the dataset's
    actual raw column names). Do NOT hardcode "observation.state"/"action"
    here as if they were universal -- that's kobo's raw column naming only;
    e.g. Libero's raw columns are "state"/"actions" instead (confirmed via a
    real local smoke test against physical-intelligence/libero -- see
    scripts/qc_label_rewards.py's module docstring for the dataset-agnostic
    pipeline this feeds)."""
    repack = next(t for t in data_config.repack_transforms.inputs if isinstance(t, _transforms.RepackTransform))
    return repack.structure["observation/state"], repack.structure["actions"]


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
        self._state_key, self._action_key = _infer_raw_state_action_keys(self.data_config)

    def __len__(self):
        return self.approx_valid_samples

    def _build_dataset(self) -> LeRobotDataset:
        step = 1.0 / self.fps
        delta_timestamps = {
            self._state_key: [0.0, step],
            self._action_key: [i * step for i in range(self.action_horizon)],
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

        state = np.asarray(item[self._state_key])
        state_t, state_t1 = state[0].astype(np.float32), state[1].astype(np.float32)
        action = np.asarray(item[self._action_key], dtype=np.float32)
        task = item.get("task", "")
        if not isinstance(task, str):
            task = str(task)

        # --- obs_t: full pipeline, real action chunk, real prompt ---------- #
        # "prompt" must be present BEFORE repack_transforms runs, not injected
        # after: KoboDataConfig's repack structure doesn't reference "prompt"
        # at all (extra dict keys are harmless -- RepackTransform only reads
        # keys listed in its structure), but LeRobotLiberoDataConfig's repack
        # structure DOES map "prompt": "prompt", i.e. it looks the key up
        # immediately -- found via a real local-Libero smoke test, which
        # KeyError'd here before this was moved earlier.
        raw_t = {**imgs_t, self._state_key: state_t, self._action_key: action, "prompt": task}
        data_t = self._pre_prompt(dict(raw_t))
        if "prompt" not in data_t:
            data_t["prompt"] = task
        data_t = self._post_prompt_full(data_t)
        self._ensure_image_mask(data_t)

        # --- obs_t1: reduced pipeline, dummy action (discarded), no tokenization --- #
        # RepackTransform's structure unconditionally looks up the action key
        # (and, for Libero-style configs, "prompt" too), so placeholders are
        # required even though we never use obs_t1's "actions"/prompt output
        # downstream.
        raw_t1 = {**imgs_t1, self._state_key: state_t1, self._action_key: np.zeros_like(action), "prompt": task}
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
        # LeRobotV3TransitionIterableDataset already built the real DataConfig
        # (via config.data.create(...), see _build_transform_stages) to derive
        # its transform pipeline -- reuse it here rather than rebuilding, so
        # data_config() (required by the openpi.training.data_loader.DataLoader
        # Protocol that checkpoints.save_state's save_assets callback expects)
        # returns the exact same norm_stats/asset_id the transforms were built from.
        self._data_config = dataset.data_config
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

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[tuple[_model.Observation, jnp.ndarray, _model.Observation]]:
        while True:
            item = self._queue.get()
            if isinstance(item, Exception):
                raise RuntimeError("JepaTransitionDataLoader background thread failed") from item
            yield item

    def close(self):
        self._stop.set()


# --------------------------------------------------------------------------- #
# 6. Q-chunking: (embed_t, proprio_t, action_chunk[h], chunk_reward,
#    embed_{t+h}, proprio_{t+h}, chunk_mask) for chunked-TD critic training
#    (see src/qc/). Both reward AND the vision embedding are read from a
#    cache produced once by scripts/qc_label_rewards.py (per-frame pooled
#    JEPA vision embedding + per-step JEPA prediction-error reward for the
#    whole dataset) -- critic training never touches the image pipeline or
#    the (multi-GB) JEPA/VLA model at all, only this small cache, raw
#    low-dim state/action (cheap, no video decoding), and the critic itself.
#    Reward is discount-accumulated over the chunk here (not in the cache),
#    so horizon_length/discount can change without re-running the labeling
#    pass.
# --------------------------------------------------------------------------- #

# Non-state/action columns QChunkTransitionDataset always reads off a
# dataset item -- see _get_slim_hf_dataset's docstring for why this list
# matters. The state/action column names themselves are dataset-specific
# (see _infer_raw_state_action_keys) and passed in separately.
_QCHUNK_FIXED_COLUMNS = ("episode_index", "index", "timestamp")


def _get_slim_hf_dataset(dataset: LeRobotDataset, state_key: str, action_key: str):
    """Returns a column-selected view of dataset.hf_dataset containing only
    _QCHUNK_FIXED_COLUMNS plus state_key/action_key. Indexing the FULL
    hf_dataset (dataset.hf_dataset[idx], which is what LeRobotDataset.__getitem__
    and this file's earlier "_get_item_no_video" both did) decodes every
    column present on that row -- including "observation.images.cam1"/"cam2",
    which are HuggingFace Image(decode=True) feature columns embedded
    directly in the arrow table, NOT lazily-decoded video files. That decode
    happens unconditionally at row-index time, before you ever get to pick
    which keys to keep -- so it happened regardless of delta_timestamps only
    listing state/action, and regardless of never reading the image keys off
    the result. Measured: ~237ms/item through the full dataset vs ~0ms/item
    through this select_columns()'d view (dataset[i] and a first attempt at
    skipping just _query_videos were both ~237ms -- the video branch was
    never the bottleneck; this was). Call once per LeRobotDataset instance
    (e.g. alongside _build_dataset), not per __getitem__ -- select_columns
    builds a new view each call."""
    dataset._ensure_hf_dataset_loaded()
    cols = [c for c in (*_QCHUNK_FIXED_COLUMNS, state_key, action_key) if c in dataset.hf_dataset.column_names]
    return dataset.hf_dataset.select_columns(cols)


def _get_item_no_video(dataset: LeRobotDataset, slim_hf_dataset, idx: int) -> dict:
    """Replicates LeRobotDataset.__getitem__'s non-video logic, but indexes
    slim_hf_dataset (see _get_slim_hf_dataset) instead of dataset.hf_dataset
    directly, and replicates _query_hf_dataset's column-stacking against that
    same slim view instead of calling the bound method (which reads
    self.hf_dataset -- the full, image-carrying table -- internally). Safe
    here because QChunkTransitionDataset only ever reads the raw state/action
    columns off the returned item."""
    item = slim_hf_dataset[idx]
    ep_idx = int(item["episode_index"])
    abs_idx = int(item["index"])
    if dataset.delta_indices is not None:
        query_indices, padding = dataset._get_query_indices(abs_idx, ep_idx)
        item = {**item, **padding}
        for key, q_idx in query_indices.items():
            if key not in slim_hf_dataset.column_names:
                continue  # video keys and anything else _get_slim_hf_dataset excludes
            relative_indices = (
                q_idx if dataset._absolute_to_relative_idx is None
                else [dataset._absolute_to_relative_idx[i] for i in q_idx]
            )
            column = slim_hf_dataset[key]
            item[key] = [column[j] for j in relative_indices]
    return item


class QChunkTransitionDataset(IterableDataset):
    """Streams (embed_t, proprio_t, action_chunk, chunk_reward, embed_th,
    proprio_th, chunk_mask) samples, shuffled at the episode level.
    h = horizon_length. Does NOT use LeRobotV3TransitionIterableDataset's
    image/transform pipeline at all -- see module docstring above."""

    def __init__(
        self,
        config: _config.TrainConfig,
        repo_id_or_root: str,
        horizon_length: int,
        qc_cache_path: str,
        discount: float = 0.99,
        is_local_root: bool = False,
        shuffle_episodes: bool = True,
        seed: int = 0,
    ):
        self.config = config
        self.repo_id_or_root = repo_id_or_root
        self.horizon_length = horizon_length
        self.discount = discount
        self.is_local_root = is_local_root
        self.shuffle_episodes = shuffle_episodes
        self.seed = seed

        meta = _load_meta(repo_id_or_root, is_local_root)
        self.fps = meta.fps
        self.episode_ranges = _episode_frame_ranges(meta)
        self.num_episodes = len(self.episode_ranges)

        # See LeRobotV3TransitionIterableDataset's identical use of this --
        # only built to read off the raw state/action column names, then
        # discarded (this class does NOT use the transform pipeline itself).
        data_config = config.data.create(config.assets_dirs, config.model)
        self._state_key, self._action_key = _infer_raw_state_action_keys(data_config)

        cache = np.load(qc_cache_path)
        self._rewards_by_episode = {
            int(k.split("_")[1]): cache[k] for k in cache.files if k.startswith("episode_")
        }
        self._embeds_by_episode = {
            int(k.split("_")[1]): cache[k] for k in cache.files if k.startswith("embed_")
        }
        # [n_frames, num_candidates, horizon_length, action_dim] per episode --
        # candidate action chunks sampled from the FROZEN BC actor at labeling
        # time (scripts/qc_label_rewards.py), used as the Q-chunking TD
        # target's off-policy "next actions" (scored by the critic being
        # trained, in src/qc/train_step.py) instead of the actual recorded
        # next action chunk (Phase 1's SARSA-style simplification). Requires
        # this cache to have been built with the SAME horizon_length as here.
        self._candidates_by_episode = {
            int(k.split("_")[1]): cache[k] for k in cache.files if k.startswith("candidates_")
        }
        if "_horizon_length" in cache.files and int(cache["_horizon_length"]) != horizon_length:
            raise ValueError(
                f"qc_cache_path was built with horizon_length={int(cache['_horizon_length'])}, "
                f"but this dataset was constructed with horizon_length={horizon_length} -- "
                "candidate action chunks won't match. Re-run qc_label_rewards.py with "
                "--horizon-length matching this value."
            )
        missing = [i for i in range(self.num_episodes) if i not in self._rewards_by_episode]
        if missing:
            raise ValueError(
                f"qc_cache_path ({qc_cache_path}) is missing {len(missing)} episodes present "
                f"in the dataset (e.g. {missing[:5]}) -- was it computed against this same dataset root, "
                "with scripts/qc_label_rewards.py?"
            )

    def _build_dataset(self) -> LeRobotDataset:
        # Only the raw state/action columns are ever accessed by _get_chunk
        # below -- no camera columns in delta_timestamps, and image columns
        # are never indexed on the returned item. See _get_slim_hf_dataset's
        # docstring for why this dataset's per-item access still needs to go
        # through a column-selected view rather than plain dataset[idx]/
        # dataset.hf_dataset[idx] to actually avoid image decode overhead.
        step = 1.0 / self.fps
        chunk_span = self.horizon_length * step
        delta_timestamps = {
            self._state_key: [0.0, chunk_span],
            self._action_key: [i * step for i in range(self.horizon_length)],
        }
        kwargs = {"root": self.repo_id_or_root, "repo_id": self.repo_id_or_root} if self.is_local_root else {"repo_id": self.repo_id_or_root}
        return LeRobotDataset(delta_timestamps=delta_timestamps, **kwargs)

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        if getattr(self, "_cached_dataset", None) is None:
            self._cached_dataset = self._build_dataset()
            self._cached_slim_hf = _get_slim_hf_dataset(self._cached_dataset, self._state_key, self._action_key)
        dataset = self._cached_dataset
        slim_hf_dataset = self._cached_slim_hf

        rng = np.random.default_rng(self.seed + worker_id)
        episode_order = np.arange(self.num_episodes)
        if self.shuffle_episodes:
            rng.shuffle(episode_order)
        episode_order = episode_order[worker_id::num_workers]

        for ep_i in episode_order:
            frm, to = self.episode_ranges[int(ep_i)]
            ep_rewards = self._rewards_by_episode[int(ep_i)]
            # qc_label_rewards.py caches embeddings/candidates for the same
            # n=len(ep_rewards) frames as rewards (obs_t of each of the n
            # valid transitions) -- one short of all (to-frm) frames in the
            # episode (the very last frame's embedding is never cached). A
            # chunk starting at local index i needs embed[i] and
            # embed[i+horizon_length] (+ candidates[i+horizon_length] for the
            # TD target) all valid, i.e. i in [0, n_valid).
            n_valid = len(ep_rewards) - self.horizon_length
            if n_valid <= 0:
                continue
            local_indices = np.arange(n_valid)
            if self.shuffle_episodes:
                rng.shuffle(local_indices)
            ep_embeds = self._embeds_by_episode[int(ep_i)]
            ep_candidates = self._candidates_by_episode[int(ep_i)]
            for local_idx in local_indices:
                global_idx = frm + int(local_idx)
                yield self._get_chunk(
                    dataset, slim_hf_dataset, global_idx, int(local_idx), ep_rewards, ep_embeds, ep_candidates
                )

    def _get_chunk(
        self,
        dataset: LeRobotDataset,
        slim_hf_dataset,
        real_idx: int,
        local_idx: int,
        ep_rewards: np.ndarray,
        ep_embeds: np.ndarray,
        ep_candidates: np.ndarray,
    ) -> dict:
        item = _get_item_no_video(dataset, slim_hf_dataset, real_idx)

        state = np.asarray(item[self._state_key])
        proprio_t, proprio_th = state[0].astype(np.float32), state[1].astype(np.float32)
        action_chunk = np.asarray(item[self._action_key], dtype=np.float32)  # [horizon_length, action_dim]

        # Off-policy TD target candidates: num_candidates action chunks
        # sampled from the FROZEN BC actor at obs_{t+h} (cached once at
        # labeling time, since the BC actor never changes during critic
        # training -- see scripts/qc_label_rewards.py's module docstring).
        # src/qc/train_step.py's critic_loss_fn scores these with the target
        # critic and picks the best, matching the reference's best-of-N target
        # (acfql.py's actor_type="best-of-n") without ever needing the VLA
        # model loaded during critic training.
        next_action_candidates = ep_candidates[local_idx + self.horizon_length]  # [num_candidates, h, a]

        # Discounted cumulative reward over the chunk: r0 + gamma*r1 + ... +
        # gamma^(h-1)*r_{h-1}, matching the reference repo's sample_sequence
        # semantics (confirmed from ColinQiyangLi/qc's utils/datasets.py).
        # mask is always 1.0 here: local_idx only ever ranges over [0, n_valid)
        # (see __iter__), which by construction never crosses an episode
        # boundary -- kobo's offline demo episodes have no early-termination
        # events mid-episode. Kept as an explicit field for compatibility with
        # the critic's expected batch keys (acfql.py's critic_loss consumes
        # batch['masks'][..., -1]) and in case kobo data later gets per-frame
        # success/failure labels that would make it meaningfully non-trivial.
        step_rewards = ep_rewards[local_idx : local_idx + self.horizon_length]
        discounts = self.discount ** np.arange(self.horizon_length)
        chunk_reward = np.float32(np.sum(step_rewards * discounts))
        chunk_mask = np.float32(1.0)

        return {
            "embed_t": ep_embeds[local_idx],
            "proprio_t": proprio_t,
            "action_chunk": action_chunk,
            "reward": chunk_reward,
            "embed_th": ep_embeds[local_idx + self.horizon_length],
            "proprio_th": proprio_th,
            "next_action_candidates": next_action_candidates,
            "mask": chunk_mask,
        }


def _collate_qchunk(batch: list[dict]) -> dict:
    return {k: np.stack([b[k] for b in batch]) for k in batch[0]}


def raw_batch_to_qchunk(
    raw: dict,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        jnp.asarray(raw["embed_t"]),
        jnp.asarray(raw["proprio_t"]),
        jnp.asarray(raw["action_chunk"]),
        jnp.asarray(raw["reward"]),
        jnp.asarray(raw["embed_th"]),
        jnp.asarray(raw["proprio_th"]),
        jnp.asarray(raw["next_action_candidates"]),
        jnp.asarray(raw["mask"]),
    )


# --------------------------------------------------------------------------- #
# 7. JAX-facing infinite iterator for Q-chunking -- mirrors JepaTransitionDataLoader.
# --------------------------------------------------------------------------- #

class QChunkDataLoader:
    """No data_config()/openpi.training.checkpoints integration -- unlike
    JepaTransitionDataLoader, this doesn't run the Observation-based transform
    pipeline at all (see QChunkTransitionDataset above), so there's no
    DataConfig/norm_stats/asset_id to expose. The critic's own checkpointing
    (scripts/train_qc_critic.py) doesn't reuse openpi.training.checkpoints'
    save_state, which requires that Protocol -- it's a simpler, separate
    subsystem."""

    def __init__(
        self,
        config: _config.TrainConfig,
        repo_id_or_root: str,
        horizon_length: int,
        qc_cache_path: str,
        data_sharding: jax.sharding.NamedSharding,
        batch_size: int,
        discount: float = 0.99,
        num_workers: int = 4,
        is_local_root: bool = False,
        prefetch: int = 4,
        seed: int = 0,
    ):
        self.data_sharding = data_sharding
        dataset = QChunkTransitionDataset(
            config, repo_id_or_root, horizon_length, qc_cache_path, discount=discount,
            is_local_root=is_local_root, shuffle_episodes=True, seed=seed,
        )
        self._torch_loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=_collate_qchunk,
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
                    batch = raw_batch_to_qchunk(raw)
                    batch = tuple(jax.device_put(x, self.data_sharding) for x in batch)
                    self._queue.put(batch)
        except Exception as e:  # noqa: BLE001
            self._queue.put(e)

    def __iter__(
        self,
    ) -> Iterator[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        while True:
            item = self._queue.get()
            if isinstance(item, Exception):
                raise RuntimeError("QChunkDataLoader background thread failed") from item
            yield item

    def close(self):
        self._stop.set()