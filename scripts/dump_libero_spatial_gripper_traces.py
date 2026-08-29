"""Dumps raw gripper-width traces around detected "release" events for
libero_spatial episodes, to distinguish genuine brief releases (real subgoal
signal) from noisy width fluctuations near the threshold boundary (a
hysteresis/threshold-calibration problem).

Follow-up to scripts/subgoal_per_suite_breakdown.py, which found
libero_spatial (a single-subgoal-by-design suite) firing MORE mid-episode
"extra" release events (mean_extra=0.579, pct_zero=45.1%) than libero_10
(genuinely multi-subgoal, mean_extra=0.517) -- the opposite of what the
detector's design assumes. This script inspects a handful of libero_spatial
episodes' actual width traces around each detected event to tell which
explanation is right.

Detection logic (open/closed thresholds, hysteresis, min-closed-frames debounce)
is duplicated from scripts/add_subgoal_reward_to_qc_cache.py's
_gripper_width/_detect_release_transitions rather than imported, since scripts/
files are run standalone (uv run scripts/<this file>.py) and are not a package
-- keep any threshold/logic changes there in sync with this file if the
heuristic itself is revised.

Usage (run from the cluster checkout, repo root):
    uv run scripts/dump_libero_spatial_gripper_traces.py \
        --dataset-root /path/to/libero \
        [--state-key state] [--open-threshold 0.06] [--closed-threshold 0.02] \
        [--min-closed-frames 10] [--n-episodes 5] [--window 15]
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "third_party/libero")
from libero.libero import benchmark  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from utils.data_loader import _episode_frame_ranges, _get_slim_hf_dataset, _load_meta  # noqa: E402

# LIBERO/robosuite convention (examples/libero/main.py): state =
# [eef_pos(3), axisangle(3), gripper_qpos(2)] -- see
# add_subgoal_reward_to_qc_cache.py's _gripper_width docstring for why this is
# a DIFFERENCE, not a sum (the two finger qpos values are opposite-signed).
GRIPPER_DIM_1 = 6
GRIPPER_DIM_2 = 7


def _gripper_width(state: np.ndarray) -> np.ndarray:
    return state[:, GRIPPER_DIM_1] - state[:, GRIPPER_DIM_2]


def _detect_release_transitions(
    width: np.ndarray, open_threshold: float, closed_threshold: float, min_closed_frames: int
) -> list[int]:
    is_closed = width < closed_threshold
    is_open = width > open_threshold
    events = []
    closed_run = 0
    armed = False
    for t in range(len(width)):
        if is_closed[t]:
            closed_run += 1
            if closed_run >= min_closed_frames:
                armed = True
        elif is_open[t]:
            if armed:
                events.append(t - 1)
                armed = False
            closed_run = 0
    return events


def main(
    dataset_root: str,
    state_key: str,
    open_threshold: float,
    closed_threshold: float,
    min_closed_frames: int,
    n_episodes: int,
    window: int,
) -> None:
    meta = _load_meta(dataset_root, is_local_root=True)
    episode_ranges = _episode_frame_ranges(meta)

    bm_dict = benchmark.get_benchmark_dict()
    spatial_tasks = {bm_dict["libero_spatial"]().get_task(i).language for i in range(bm_dict["libero_spatial"]().n_tasks)}

    tasks_col = meta.episodes["tasks"]
    spatial_episodes = [ep_i for ep_i, tasks in enumerate(tasks_col) if tasks and tasks[0] in spatial_tasks]
    print(f"Found {len(spatial_episodes)} libero_spatial episodes total.")

    dataset = LeRobotDataset(root=dataset_root, repo_id=dataset_root)
    # action_key deliberately set to a column name that doesn't exist --
    # _get_slim_hf_dataset filters to columns actually present, so this just
    # drops the action column from the slim view (we only need state here)
    # without duplicating state_key in the select_columns() call.
    slim_hf = _get_slim_hf_dataset(dataset, state_key, "__unused__")

    dumped = 0
    for ep_i in spatial_episodes:
        if dumped >= n_episodes:
            break
        frm, to = episode_ranges[ep_i]
        rows = slim_hf[frm:to]
        state = np.asarray(rows[state_key], dtype=np.float32)
        width = _gripper_width(state)
        events = _detect_release_transitions(width, open_threshold, closed_threshold, min_closed_frames)
        if not events:
            continue

        dumped += 1
        task_str = tasks_col[ep_i][0] if tasks_col[ep_i] else "<unknown>"
        print(f"\n=== episode {ep_i} ({len(width)} frames) -- task: {task_str!r} ===")
        print(f"{len(events)} detected event(s) at transition index(es): {events}")
        for e in events:
            lo = max(0, e - window)
            hi = min(len(width), e + window + 2)
            print(f"\n  -- event at t={e} (open crossing from t={e} to t={e + 1}) --")
            print(f"  {'t':>5} {'width':>10} {'state':>7}")
            for t in range(lo, hi):
                if width[t] < closed_threshold:
                    state_label = "closed"
                elif width[t] > open_threshold:
                    state_label = "open"
                else:
                    state_label = "-"
                marker = "  <== event" if t == e or t == e + 1 else ""
                print(f"  {t:>5} {width[t]:>10.4f} {state_label:>7}{marker}")

    if dumped == 0:
        print("No libero_spatial episodes with a detected mid-episode event were found in the first pass.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--state-key", default="state")
    parser.add_argument("--open-threshold", type=float, default=0.06)
    parser.add_argument("--closed-threshold", type=float, default=0.02)
    parser.add_argument("--min-closed-frames", type=int, default=10)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--window", type=int, default=15)
    args = parser.parse_args()
    main(
        args.dataset_root,
        args.state_key,
        args.open_threshold,
        args.closed_threshold,
        args.min_closed_frames,
        args.n_episodes,
        args.window,
    )
