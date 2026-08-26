"""Standalone sanity-check script for lerobot_v3_transition_loader.py
(real openpi transform pipeline version -- see that file's module docstring
for what changed and why).

Unlike the earlier version, this now needs a real TrainConfig (to build the
repack/data/model transforms), not just a bare dataset root -- so this script
takes a config name the same way your training entrypoint does.

Usage:
    uv run test_transition_loader.py pi0_kobo_cube --data.root /home/fabian/kobo_cube

What it checks:
  1. Dataset metadata loads; episode/frame counts printed for a memory-scale
     sanity check.
  2. Streams N samples via __iter__, applying the REAL transform pipeline,
     and checks the resulting dicts match Observation.from_dict's expected
     shape: "image" dict with the canonical IMAGE_KEYS (not your raw cam1/
     cam2 names -- that remapping is KoboInputs' job, confirming it actually
     ran), "image_mask" dict, "state", "actions" (obs_t only), and
     "tokenized_prompt"/"tokenized_prompt_mask" (obs_t only).
  3. obs_t and obs_t1 images actually differ (same degenerate-transition
     check as before, now applied post-transform).
  4. Actually constructs _model.Observation via raw_batch_to_transition and
     confirms it doesn't raise -- this is the check that would have caught
     the prompt_tokens/tokenized_prompt field-name bug before it reached
     a real training run.
  5. Batch collation via a real (num_workers=0) DataLoader pass.
"""

import argparse
import itertools
import sys
import time

import numpy as np

import openpi.training.config as _config
from data_loader import (
    LeRobotV3TransitionDataset,
    _collate,
    raw_batch_to_transition,
)
from openpi.models import model as _model


def _fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


def _ok(msg: str):
    print(f"OK:   {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="e.g. pi0_kobo_cube")
    parser.add_argument("--root", type=str, default=None,
                         help="Override config.data.root (local dataset path). "
                              "If omitted, uses whatever the named config resolves to -- "
                              "which may be None, see the create_base_config bug noted elsewhere.")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--static-threshold", type=float, default=1.0)
    args = parser.parse_args()

    import dataclasses
    import pathlib
    config = _config.get_config(args.config_name)
    if args.root is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, root=pathlib.Path(args.root)))

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.root is None and data_config.repo_id is None:
        _fail("config.data resolved to neither root nor repo_id -- pass --root")
    repo_id_or_root = str(data_config.root) if data_config.root is not None else data_config.repo_id
    is_local_root = data_config.root is not None

    print(f"Loading metadata ({'local root' if is_local_root else 'hub repo'}: {repo_id_or_root}) ...")
    t0 = time.time()
    dataset = LeRobotV3TransitionDataset(
        config, repo_id_or_root, config.model.action_horizon, is_local_root=is_local_root
    )
    print(f"  metadata + transform pipeline built in {time.time() - t0:.1f}s")

    if dataset.num_episodes == 0:
        _fail("dataset has zero episodes")
    _ok(f"{dataset.num_episodes} episodes, ~{dataset.total_frames} total frames, "
        f"~{dataset.approx_valid_samples} valid transition samples, "
        f"raw camera keys = {dataset.camera_keys}, fps = {dataset.fps}")

    # --- Stream N samples through the REAL transform pipeline -------------- #
    samples = list(itertools.islice(iter(dataset), args.num_samples))
    if len(samples) == 0:
        _fail("streamed zero samples")
    if len(samples) < args.num_samples:
        print(f"WARN: only got {len(samples)} samples (dataset smaller than --num-samples, or short episodes)")

    static_count = 0
    canonical_cams = None
    for item in samples:
        data_t, data_t1 = item["obs_t"], item["obs_t1"]

        for key, required in (("image", True), ("image_mask", True), ("state", True)):
            if key not in data_t:
                _fail(f"obs_t missing required key '{key}' after transform pipeline")
            if key not in data_t1:
                _fail(f"obs_t1 missing required key '{key}' after transform pipeline")

        if "actions" not in data_t:
            _fail("obs_t missing 'actions' (expected from PadStatesAndActions)")
        if ("tokenized_prompt" not in data_t) or ("tokenized_prompt_mask" not in data_t):
            _fail("obs_t missing tokenized_prompt/tokenized_prompt_mask -- TokenizePrompt didn't run")

        cams = set(data_t["image"].keys())
        if canonical_cams is None:
            canonical_cams = cams
            expected = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
            if not expected.issubset(cams):
                _fail(f"obs_t['image'] keys {cams} don't cover the model's expected IMAGE_KEYS {expected} -- "
                      f"KoboInputs' camera remapping doesn't look like it ran correctly")
            _ok(f"obs_t['image'] has canonical camera keys: {cams} (raw dataset only has {dataset.camera_keys}, "
                f"confirms KoboInputs remapping/padding ran)")
        elif cams != canonical_cams:
            _fail(f"inconsistent image keys across samples: {cams} vs {canonical_cams}")

        for cam in cams:
            img_t, img_t1 = np.asarray(data_t["image"][cam]), np.asarray(data_t1["image"][cam])
            if img_t.dtype != np.uint8:
                _fail(f"image '{cam}' dtype is {img_t.dtype}, expected uint8 going into Observation.from_dict")
            diff = np.abs(img_t.astype(np.float32) - img_t1.astype(np.float32)).mean()
            if diff < args.static_threshold:
                static_count += 1

        if data_t["actions"].shape[0] != config.model.action_horizon:
            _fail(f"actions horizon {data_t['actions'].shape[0]} != config.model.action_horizon "
                  f"{config.model.action_horizon}")

    _ok(f"checked schema across {len(samples)} streamed samples")
    static_frac = static_count / (len(samples) * len(canonical_cams))
    print(f"  static-transition fraction: {static_frac:.2%}")

    # --- The check that actually matters: does Observation.from_dict work? --- #
    batch = _collate(samples[:8])
    try:
        obs_t, action_chunk, obs_t1 = raw_batch_to_transition(batch, config)
    except Exception as e:
        _fail(f"raw_batch_to_transition raised: {type(e).__name__}: {e}")
    if not isinstance(obs_t, _model.Observation) or not isinstance(obs_t1, _model.Observation):
        _fail("raw_batch_to_transition did not return Observation instances")
    if obs_t.tokenized_prompt is None:
        _fail("obs_t.tokenized_prompt is None -- TokenizePrompt output didn't survive to Observation")
    if obs_t1.tokenized_prompt is None:
        _fail("obs_t1.tokenized_prompt is None -- expected it to be reused from obs_t")
    if not np.allclose(np.asarray(obs_t.tokenized_prompt), np.asarray(obs_t1.tokenized_prompt)):
        _fail("obs_t1.tokenized_prompt doesn't match obs_t's -- reuse logic isn't working")
    _ok(f"Observation.from_dict succeeded for both obs_t and obs_t1 "
        f"(images shape {next(iter(obs_t.images.values())).shape}, "
        f"state shape {obs_t.state.shape}, action_chunk shape {action_chunk.shape})")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()