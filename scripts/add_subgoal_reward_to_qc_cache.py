#!/usr/bin/env python3
"""Smarter version of add_goal_reward_to_qc_cache.py's sparse terminal reward:
many LIBERO tasks are actually multiple subgoals in one episode (e.g. "put
BOTH the alphabet soup and the tomato sauce in the basket," "put the white
mug on the left plate AND put the yellow/white mug on the right plate") --
rewarding only the very last transition ignores every earlier subgoal
completion, which is exactly the kind of task-progress signal we want the
critic to actually learn from.

Detects subgoal completions from data alone (no simulator/ground-truth
object-pose access needed at labeling time) via a GRIPPER RELEASE heuristic:
LIBERO's pick-and-place-style tasks complete a subgoal by opening the gripper
after having held it closed for a while (placing/releasing an object). State
layout (see examples/libero/main.py's element construction, robosuite/LIBERO's
standard convention): observation.state = [eef_pos(3), axisangle(3),
gripper_qpos(2)] -- gripper_qpos are the two finger positions (both near 0 =
closed/holding, both near their max = open), so "width" = qpos[0]+qpos[1] is
a clean scalar open/closed signal, with hysteresis thresholds to ignore noise
and a minimum-closed-duration debounce so tiny regrasp jitter doesn't count
as a "release."

ASSUMPTIONS (same as add_goal_reward_to_qc_cache.py, read that docstring
too):
  - every episode is a SUCCESSFUL demonstration
  - observation.state's layout matches LIBERO's standard 8-dim convention
    above -- if this dataset's state column means something else, this
    script's gripper-width extraction is wrong and should be fixed before
    trusting its output.

The final transition of every episode ALWAYS gets rewarded too (regardless
of whether gripper-release detection fires there), since episode-end is
definitionally a successful completion even for tasks that don't end with an
open gripper (e.g. insertion/pushing tasks).

Run with --dry-run first: prints per-episode detected event counts and
gripper-width distribution stats WITHOUT writing anything, so you can
sanity-check the thresholds actually separate "open" from "closed" for this
real dataset before trusting them on the full run (defaults are robosuite/
Panda-gripper-typical values, not calibrated against this specific dataset).

Usage:
    uv run scripts/add_subgoal_reward_to_qc_cache.py <config_name> \
        --data.root=/path/to/dataset \
        --input-path=/path/to/qc_cache.npz \
        --output-path=/path/to/qc_cache_with_subgoals.npz \
        [--dry-run] [--open-threshold=0.06] [--closed-threshold=0.02] \
        [--min-closed-frames=10] [--reward-value=1.0]
"""

import argparse
import logging
import sys

import numpy as np

import openpi.training.config as _config
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils.data_loader import (
    _episode_frame_ranges,
    _get_slim_hf_dataset,
    _infer_raw_state_action_keys,
    _load_meta,
    resolve_dataset_root,
)

# LIBERO/robosuite convention (examples/libero/main.py): state =
# [eef_pos(3), axisangle(3), gripper_qpos(2)] -- the last two dims are the
# two-finger gripper positions.
GRIPPER_DIM_1 = 6
GRIPPER_DIM_2 = 7


def _gripper_width(state: np.ndarray) -> np.ndarray:
    """state: [T, state_dim] -> [T] scalar "how open is the gripper" signal.

    Confirmed directly against a real converted LIBERO episode (NOT assumed):
    the two finger qpos values are opposite-signed and near-symmetric around
    zero (finger1 in ~[0.002, 0.04], finger2 in ~[-0.04, -0.005]) -- a naive
    sum() cancels to ~0 regardless of open/closed state (that WAS this
    function's first version, and it made --dry-run report zero release
    events anywhere, since nothing could ever cross a positive open
    threshold). The DIFFERENCE (dim1 - dim2) is what actually tracks
    open/closed: ~0.08 fully open, shrinking toward 0 as the gripper closes.
    """
    return state[:, GRIPPER_DIM_1] - state[:, GRIPPER_DIM_2]


def _detect_release_transitions(
    width: np.ndarray, open_threshold: float, closed_threshold: float, min_closed_frames: int
) -> list[int]:
    """Returns transition-indices (0-indexed into a length-(T-1) reward array)
    where the gripper opened after being closed for >= min_closed_frames
    consecutive frames -- a proxy for "just released a held object,"
    hopefully corresponding to a subgoal completion. Hysteresis (values
    between closed_threshold and open_threshold count as neither) avoids
    threshold-boundary noise flipping state back and forth."""
    is_closed = width < closed_threshold
    is_open = width > open_threshold
    events = []
    closed_run = 0
    armed = False  # True once we've been closed long enough that an open counts as a real release
    for t in range(len(width)):
        if is_closed[t]:
            closed_run += 1
            if closed_run >= min_closed_frames:
                armed = True
        elif is_open[t]:
            if armed:
                events.append(t - 1)  # transition FROM frame t-1 TO frame t
                armed = False
            closed_run = 0
        # else: dead zone between thresholds -- hold closed_run/armed state as-is
    return events


def _build_dataset(repo_id_or_root: str, is_local_root: bool) -> LeRobotDataset:
    kwargs = {"root": repo_id_or_root, "repo_id": repo_id_or_root} if is_local_root else {"repo_id": repo_id_or_root}
    return LeRobotDataset(**kwargs)


def main(
    config: _config.TrainConfig,
    input_path: str,
    output_path: str,
    open_threshold: float,
    closed_threshold: float,
    min_closed_frames: int,
    reward_value: float,
    dry_run: bool,
) -> None:
    # basicConfig alone is a no-op here: importing openpi.training.config
    # already installs a root StreamHandler (confirmed directly) at whatever
    # level it defaults to (WARNING), so INFO messages below would otherwise
    # be silently dropped despite exiting 0 -- explicit setLevel is required.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger().setLevel(logging.INFO)

    cache = np.load(input_path)
    episode_keys = sorted(
        (k for k in cache.files if k.startswith("episode_")), key=lambda k: int(k.split("_")[1])
    )
    if not episode_keys:
        raise RuntimeError(f"No episode_<i> reward arrays found in {input_path} -- wrong cache file?")

    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id_or_root, is_local_root = resolve_dataset_root(data_config)
    state_key, _action_key = _infer_raw_state_action_keys(data_config)

    dataset = _build_dataset(repo_id_or_root, is_local_root)
    slim_hf = _get_slim_hf_dataset(dataset, state_key, _action_key)
    meta = _load_meta(repo_id_or_root, is_local_root)
    episode_ranges = _episode_frame_ranges(meta)

    all_widths = []
    event_counts = []
    out = {k: cache[k] for k in cache.files}
    for key in episode_keys:
        ep_i = int(key.split("_", 1)[1])
        n = len(cache[key])  # reward array length == transitions == frames - 1
        if n <= 0:
            out[f"goal_{ep_i}"] = np.zeros(0, dtype=np.float32)
            continue

        frm, to = episode_ranges[ep_i]
        rows = slim_hf[frm:to]
        state = np.asarray(rows[state_key], dtype=np.float32)  # [frames, state_dim]
        width = _gripper_width(state)
        all_widths.append(width)

        events = _detect_release_transitions(width, open_threshold, closed_threshold, min_closed_frames)
        events = [e for e in events if 0 <= e < n]

        goal = np.zeros(n, dtype=np.float32)
        for e in events:
            goal[e] = reward_value
        goal[n - 1] = reward_value  # episode end always counts as completion, release-detected or not
        event_counts.append(len(events))

        if not dry_run:
            out[f"goal_{ep_i}"] = goal

    all_widths_flat = np.concatenate(all_widths) if all_widths else np.zeros(0)
    logging.info(
        f"Gripper width stats across {len(episode_keys)} episodes: "
        f"mean={all_widths_flat.mean():.4f} std={all_widths_flat.std():.4f} "
        f"min={all_widths_flat.min():.4f} max={all_widths_flat.max():.4f} "
        f"p10={np.percentile(all_widths_flat, 10):.4f} p90={np.percentile(all_widths_flat, 90):.4f}"
    )
    logging.info(
        f"Detected release events per episode: mean={np.mean(event_counts):.2f} "
        f"min={min(event_counts)} max={max(event_counts)} "
        f"(thresholds: closed<{closed_threshold}, open>{open_threshold}, min_closed_frames={min_closed_frames}) "
        f"-- if this doesn't look like a sane subgoal count for these tasks (LIBERO's multi-object suites should "
        f"show >1 on average, single-object suites near 1), the thresholds probably need adjusting against the "
        f"real width stats printed above before trusting the output."
    )

    if dry_run:
        logging.info("--dry-run: nothing written. Drop --dry-run once thresholds look right.")
        return

    np.savez(output_path, **out)
    logging.info(f"Wrote goal_<i> arrays (subgoal-aware) to {output_path}")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--input-path", type=str, required=True)
    _parser.add_argument("--output-path", type=str, default=None)
    _parser.add_argument("--open-threshold", type=float, default=0.06)
    _parser.add_argument("--closed-threshold", type=float, default=0.02)
    _parser.add_argument("--min-closed-frames", type=int, default=10)
    _parser.add_argument("--reward-value", type=float, default=1.0)
    _parser.add_argument("--dry-run", action="store_true", default=False)
    _args, _remaining = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    if not _args.dry_run and not _args.output_path:
        raise SystemExit("--output-path is required unless --dry-run is set")

    main(
        _config.cli(),
        _args.input_path,
        _args.output_path,
        _args.open_threshold,
        _args.closed_threshold,
        _args.min_closed_frames,
        _args.reward_value,
        _args.dry_run,
    )
