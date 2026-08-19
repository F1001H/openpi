#!/usr/bin/env python3
"""Adds a sparse task-completion ("goal") reward alongside the existing
JEPA-prediction-error intrinsic reward in an already-labeled QC cache
(scripts/qc_label_rewards.py's output), WITHOUT re-running that (expensive,
JEPA-forward-pass-per-frame) labeling pass.

WHY: every best-of-N variant tried so far (argmax, argmin, argmax + critic
Q-head disagreement, argmax + actor sample-disagreement -- see
libero_qc_*_eval_results*.txt) is trained/scored on PURE intrinsic
(predictability) reward, which carries no information about whether the
episode is actually succeeding. That's the likely root cause of the
spatial-precision failures seen under every variant: the reward has nothing
to say about "closer to correctly completing the task," only "how surprising
was this transition." Combining intrinsic reward with a sparse GOAL reward
(+1 at the last transition of each episode) gives the critic something to
propagate backward via its existing discounted TD bootstrap (src/qc/
train_step.py) that's actually grounded in task completion.

ASSUMPTION: every episode in the dataset is a SUCCESSFUL demonstration (true
for LIBERO's official curated demo data, which is what this project's
qc_label_rewards.py runs were labeling) -- if that's not true for some other
dataset, this sparse "+1 at the end" signal would be teaching the critic to
value reaching the end of ANY episode, not a successful one, which would be
actively wrong. Don't reuse this script on a dataset with failed/mixed-outcome
episodes without changing this.

Cheap by design: episode lengths are already implicit in the existing
episode_<i> (intrinsic reward) arrays' shapes, so this needs zero GPU/model
compute, just numpy over the cache -- runs in seconds even for the full
dataset, unlike the labeling pass itself (hours).

Usage:
    uv run scripts/add_goal_reward_to_qc_cache.py \
        --input-path=/path/to/qc_cache.npz \
        --output-path=/path/to/qc_cache_with_goal.npz \
        [--goal-value=1.0]
"""

import argparse
import logging

import numpy as np


def add_goal_reward(input_path: str, output_path: str, goal_value: float) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data = np.load(input_path)

    episode_keys = [k for k in data.files if k.startswith("episode_")]
    if not episode_keys:
        raise RuntimeError(f"No episode_<i> reward arrays found in {input_path} -- wrong cache file?")

    out = {k: data[k] for k in data.files}
    for key in episode_keys:
        n = len(data[key])
        goal = np.zeros(n, dtype=np.float32)
        if n > 0:
            # Sparse terminal reward: only the LAST transition of the
            # episode (the one whose obs_t1 is the final, successful state)
            # gets the goal bonus. Everything before it is 0 -- the
            # discounted TD bootstrap in qc/train_step.py is what propagates
            # this backward through training, not this script.
            goal[-1] = goal_value
        ep_i = key.split("_", 1)[1]
        out[f"goal_{ep_i}"] = goal

    np.savez(output_path, **out)

    all_intrinsic = np.concatenate([data[k] for k in episode_keys])
    logging.info(
        f"Wrote {len(episode_keys)} goal_<i> arrays (sparse +{goal_value} at each episode's last transition) "
        f"to {output_path}"
    )
    logging.info(
        f"For reference, existing intrinsic reward stats: mean={all_intrinsic.mean():.6f} "
        f"std={all_intrinsic.std():.6f} min={all_intrinsic.min():.6f} max={all_intrinsic.max():.6f} "
        f"-- use this to sanity-check --alpha-goal/--alpha-intrinsic's RELATIVE scale when training the critic "
        f"(train_qc_critic.py), since goal_value={goal_value} sparse is likely much larger than a typical "
        f"per-step intrinsic reward on its own."
    )


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument("--input-path", type=str, required=True)
    _parser.add_argument("--output-path", type=str, required=True)
    _parser.add_argument("--goal-value", type=float, default=1.0)
    _args = _parser.parse_args()
    add_goal_reward(_args.input_path, _args.output_path, _args.goal_value)
