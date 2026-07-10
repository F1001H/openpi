"""Standalone sanity-check script for lerobot_v3_transition_loader.py.

Not a pytest suite -- run directly against a real dataset, since the whole
point is to catch things that only show up against real data (episode
boundary handling, actual image layout, degenerate transitions), which can't
be verified with mocks in an environment that doesn't have `lerobot`/`jax`
installed.

Usage:
    python test_transition_loader.py --repo-id lerobot/aloha_sim_transfer_cube_human
    python test_transition_loader.py --root /path/to/local/dataset --local-root

What it checks:
  1. Dataset loads, has the expected camera keys, non-zero length.
  2. No sampled index is a last-frame-of-episode index (the whole reason the
     filtering exists -- if this fails, obs_t == obs_t1 degenerate transitions
     are leaking through).
  3. Shapes: images HWC uint8, state 1D, action [horizon, action_dim].
  4. obs_t and obs_t1 are actually different (catches delta_timestamps
     misconfiguration silently padding/repeating the same frame).
  5. Action horizon matches what was requested.
  6. Batch collation produces consistent shapes across the batch dimension.
  7. Reports the fraction of "suspiciously static" transitions (image barely
     changed) -- some is normal (robot paused), but a high fraction usually
     means something's wrong with the delta/timestamp alignment rather than
     the robot being idle.

Exits non-zero on any hard failure so it's usable in CI once you have a
lerobot-enabled runner.
"""

import argparse
import sys
import time

import numpy as np

from data_loader import (
    LeRobotV3TransitionDataset,
    _collate,
    _last_frame_global_indices,
)


def _fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


def _ok(msg: str):
    print(f"OK:   {msg}")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo-id", type=str, help="Hub repo id, e.g. lerobot/aloha_sim_transfer_cube_human")
    src.add_argument("--root", type=str, help="Local dataset root path")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=200, help="How many individual __getitem__ calls to spot-check")
    parser.add_argument("--static-threshold", type=float, default=1.0,
                         help="Mean abs pixel diff (0-255 scale) below which a transition is flagged 'static'")
    parser.add_argument("--static-frac-warn", type=float, default=0.5,
                         help="Warn if more than this fraction of sampled transitions are static")
    args = parser.parse_args()

    repo_id_or_root = args.root if args.root else args.repo_id
    is_local_root = args.root is not None

    print(f"Loading dataset ({'local root' if is_local_root else 'hub repo'}: {repo_id_or_root}) ...")
    t0 = time.time()
    dataset = LeRobotV3TransitionDataset(
        repo_id_or_root, args.action_horizon, is_local_root=is_local_root
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    if len(dataset) == 0:
        _fail("dataset has zero valid (non-last-frame) samples")
    _ok(f"dataset has {len(dataset)} valid transition samples, camera keys = {dataset.camera_keys}")

    # --- Check 2: no valid index is a last-frame index -------------------- #
    last_idx = set(int(i) for i in _last_frame_global_indices(dataset.dataset))
    overlap = set(int(i) for i in dataset.valid_indices) & last_idx
    if overlap:
        _fail(f"{len(overlap)} valid indices are actually last-frame-of-episode indices "
              f"(episode boundary filtering is broken): e.g. {sorted(overlap)[:5]}")
    _ok("no valid index overlaps with last-frame-of-episode indices")

    # --- Checks 3-4-5: per-sample shape + degeneracy spot check ----------- #
    n = min(args.num_samples, len(dataset))
    rng = np.random.default_rng(0)
    sample_positions = rng.choice(len(dataset), size=n, replace=False)

    static_count = 0
    action_dim = None
    state_dim = None

    for pos in sample_positions:
        item = dataset[int(pos)]

        for cam in dataset.camera_keys:
            img_t = item["images"][cam]
            img_t1 = item["next_images"][cam]
            if img_t.dtype != np.uint8:
                _fail(f"image '{cam}' dtype is {img_t.dtype}, expected uint8")
            if img_t.ndim != 3 or img_t.shape[-1] not in (1, 3):
                _fail(f"image '{cam}' shape {img_t.shape} doesn't look like HWC")
            if img_t.shape != img_t1.shape:
                _fail(f"image '{cam}' shape mismatch between t ({img_t.shape}) and t+1 ({img_t1.shape})")
            diff = np.abs(img_t.astype(np.float32) - img_t1.astype(np.float32)).mean()
            if diff < args.static_threshold:
                static_count += 1

        if item["action"].shape[0] != args.action_horizon:
            _fail(f"action horizon {item['action'].shape[0]} != requested {args.action_horizon}")
        action_dim = action_dim or item["action"].shape[1]
        if item["action"].shape[1] != action_dim:
            _fail("action_dim is inconsistent across samples")

        state_dim = state_dim or item["state"].shape[0]
        if item["state"].shape[0] != state_dim:
            _fail("state_dim is inconsistent across samples")
        if item["next_state"].shape[0] != state_dim:
            _fail("next_state dim doesn't match state dim")

    _ok(f"checked shapes/dtypes across {n} samples "
        f"(action_dim={action_dim}, state_dim={state_dim})")

    static_frac = static_count / (n * len(dataset.camera_keys))
    print(f"  static-transition fraction across sampled (sample, camera) pairs: {static_frac:.2%}")
    if static_frac > args.static_frac_warn:
        print(f"WARN: {static_frac:.2%} of sampled transitions look static (mean abs pixel diff "
              f"< {args.static_threshold}). If your data isn't mostly the robot sitting idle, this "
              f"usually means delta_timestamps is returning the same frame twice -- double-check "
              f"fps/step alignment and the episode-boundary filtering.")
    else:
        _ok(f"static-transition fraction ({static_frac:.2%}) looks reasonable")

    # --- Check 6: batch collation ----------------------------------------- #
    batch_items = [dataset[int(i)] for i in sample_positions[: args.batch_size]]
    batch = _collate(batch_items)
    for cam in dataset.camera_keys:
        if batch["images"][cam].shape[0] != len(batch_items):
            _fail(f"collated batch dim mismatch for images['{cam}']")
        if batch["next_images"][cam].shape != batch["images"][cam].shape:
            _fail(f"collated images/next_images shape mismatch for '{cam}'")
    if batch["action"].shape != (len(batch_items), args.action_horizon, action_dim):
        _fail(f"collated action shape {batch['action'].shape} != "
              f"expected {(len(batch_items), args.action_horizon, action_dim)}")
    if batch["state"].shape != (len(batch_items), state_dim):
        _fail(f"collated state shape {batch['state'].shape} != expected {(len(batch_items), state_dim)}")
    _ok(f"batch collation OK for batch_size={len(batch_items)}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()