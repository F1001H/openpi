"""Breaks down add_subgoal_reward_to_qc_cache.py's detected release events by
LIBERO suite, to check whether a reported mean-extra-events figure (per
episode, on top of the guaranteed final-frame reward) is concentrated in
libero_10 (multi-object tasks, should show ~1+ extra) vs near-zero on the
single-object suites (spatial/object/goal, where 0 extra is correct
behavior, not under-detection).

Usage (run from the cluster checkout, repo root):
    uv run scripts/subgoal_per_suite_breakdown.py \
        --cache /path/to/qc_cache_..._with_subgoal.npz \
        --dataset-root /path/to/libero
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "third_party/libero")
from libero.libero import benchmark  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402


def main(cache_path: str, dataset_root: str) -> None:
    cache = np.load(cache_path)
    meta = LeRobotDatasetMetadata(repo_id=dataset_root, root=dataset_root)

    # Build task-string -> suite lookup from LIBERO's own benchmark
    # definitions (the ground truth for suite membership, not a guess from
    # task-string patterns).
    bm_dict = benchmark.get_benchmark_dict()
    task_to_suite: dict[str, str] = {}
    for suite_name in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        suite = bm_dict[suite_name]()
        for i in range(suite.n_tasks):
            task_to_suite[suite.get_task(i).language] = suite_name

    tasks_col = meta.episodes["tasks"]
    buckets: dict[str, list[int]] = {
        "libero_spatial": [], "libero_object": [], "libero_goal": [], "libero_10": [], "unknown": [],
    }
    unknown_examples = []

    for ep_i, tasks in enumerate(tasks_col):
        task_str = tasks[0] if tasks else ""
        key = f"goal_{ep_i}"
        if key not in cache.files:
            continue
        goal = cache[key]
        nonzero = int(np.sum(goal > 0))
        # Subtract the guaranteed final-frame reward (always set, see
        # add_subgoal_reward_to_qc_cache.py: goal[n-1] = reward_value
        # unconditionally) to isolate mid-episode-detected events only.
        extra = max(0, nonzero - 1)
        suite = task_to_suite.get(task_str, "unknown")
        buckets[suite].append(extra)
        if suite == "unknown" and len(unknown_examples) < 5:
            unknown_examples.append(task_str)

    print(f"{'suite':<16} {'n_episodes':>10} {'mean_extra':>11} {'max_extra':>10} {'pct_zero':>9}")
    for suite, counts in buckets.items():
        if not counts:
            continue
        counts_arr = np.array(counts)
        pct_zero = 100.0 * np.mean(counts_arr == 0)
        print(f"{suite:<16} {len(counts):>10} {np.mean(counts_arr):>11.3f} {counts_arr.max():>10} {pct_zero:>8.1f}%")

    if unknown_examples:
        print(f"\n{len(buckets['unknown'])} episodes had a task string not matching any of the 4 official "
              f"suites' task lists -- examples: {unknown_examples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    main(args.cache, args.dataset_root)
