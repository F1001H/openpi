#!/bin/bash
#SBATCH --job-name=train_libero_qc_critic_full_finetune_subgoal
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Subgoal-aware counterpart to train_libero_qc_critic_full_finetune_goal.slurm
# (which only rewards the very last transition of each episode). Many LIBERO
# tasks have MULTIPLE subgoals per episode (e.g. "put BOTH the alphabet soup
# AND the tomato sauce in the basket") -- a single terminal reward ignores
# every earlier subgoal completion, wasting most of the task-progress signal
# actually present in the demos. scripts/add_subgoal_reward_to_qc_cache.py
# detects subgoal completions via a gripper-release heuristic (gripper opens
# after being held closed for a while = probably just placed/released an
# object) purely from the dataset's own state column -- no simulator or
# object-pose ground truth needed. See that script's docstring for the full
# mechanism and its assumptions.
#
# Three stages, all cheap (no VLA/JEPA model loaded, no re-running the
# expensive per-frame intrinsic-reward labeling pass):
#   1. --dry-run: prints real gripper-width stats + detected-events-per-
#      episode for THIS dataset, so the default thresholds (calibrated
#      against generic robosuite/Panda-gripper conventions, not this
#      specific converted dataset) can be sanity-checked in the log before
#      committing. If detected events/episode looks wrong (e.g. ~0 for
#      known multi-object tasks, or absurdly high from threshold noise),
#      stop and adjust OPEN_THRESHOLD/CLOSED_THRESHOLD/MIN_CLOSED_FRAMES
#      before rerunning.
#   2. Real run (no --dry-run): writes goal_<i> arrays into a COPY of the
#      cache (original untouched).
#   3. train_qc_critic.py --alpha-intrinsic=1.0 --alpha-goal=$ALPHA_GOAL on
#      the merged cache.
#
# This whole script is cheap enough to just rerun with different thresholds/
# ALPHA_GOAL if stage 1's output looks off -- no GPU-heavy relabeling needed.

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/libero}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
EXP_NAME="${EXP_NAME:-libero_jepa_full_finetune_critic_subgoal_beta0.1}"
INTRINSIC_CACHE_PATH="${INTRINSIC_CACHE_PATH:-$SHARED_WS/fabian/qc_cache_libero_full_finetune_beta0.1.npz}"
QC_CACHE_PATH="${QC_CACHE_PATH:-$SHARED_WS/fabian/qc_cache_libero_full_finetune_beta0.1_with_subgoal.npz}"
CRITIC_CHECKPOINT_DIR="${CRITIC_CHECKPOINT_DIR:-$SHARED_WS/fabian/checkpoints_qc/libero_full_finetune_beta0.1_subgoal}"

OPEN_THRESHOLD="${OPEN_THRESHOLD:-0.06}"
CLOSED_THRESHOLD="${CLOSED_THRESHOLD:-0.02}"
MIN_CLOSED_FRAMES="${MIN_CLOSED_FRAMES:-10}"
REWARD_VALUE="${REWARD_VALUE:-1.0}"
ALPHA_INTRINSIC="${ALPHA_INTRINSIC:-1.0}"
# Same starting-guess caveat as the plain-terminal version -- check stage 1's
# (this script's own, not the simpler script's) printed stats before trusting
# this. Likely needs to be lower than the terminal-only version's weight per
# subgoal event, since a multi-subgoal episode now gets several reward
# pulses instead of one -- the total per-episode goal signal is larger even
# at the same per-event REWARD_VALUE.
ALPHA_GOAL="${ALPHA_GOAL:-0.05}"

HORIZON_LENGTH="${HORIZON_LENGTH:-5}"
PROPRIO_DIM="${PROPRIO_DIM:-8}"
ACTION_DIM="${ACTION_DIM:-7}"
# 5, up from the reference implementation's/every prior critic's default of
# 2 -- QChunkCritic/train_step.py are already generic over num_qs (dict-of-
# heads, no double-Q-specific code), so this needed zero code changes.
# Pairs with serve_qc_policy.py --selection-mode=majority_vote (each head
# votes for its own preferred candidate, plurality wins) as a more
# principled alternative to guessing an --uncertainty-penalty weight for
# how much to distrust cross-head disagreement.
NUM_QS="${NUM_QS:-5}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$CRITIC_CHECKPOINT_DIR"

if [ ! -f "$INTRINSIC_CACHE_PATH" ]; then
    echo "ERROR: reward cache not found at $INTRINSIC_CACHE_PATH -- run slurm/label_libero_qc_rewards_full_finetune.slurm first." >&2
    exit 1
fi

echo "=== Stage 1: dry-run gripper-width/event-count sanity check ==="
uv run scripts/add_subgoal_reward_to_qc_cache.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --data.root="$DATASET_ROOT" \
    --input-path="$INTRINSIC_CACHE_PATH" \
    --open-threshold="$OPEN_THRESHOLD" \
    --closed-threshold="$CLOSED_THRESHOLD" \
    --min-closed-frames="$MIN_CLOSED_FRAMES" \
    --reward-value="$REWARD_VALUE" \
    --dry-run

echo "=== Stage 2: merging subgoal reward into a copy of the cache ==="
uv run scripts/add_subgoal_reward_to_qc_cache.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --data.root="$DATASET_ROOT" \
    --input-path="$INTRINSIC_CACHE_PATH" \
    --output-path="$QC_CACHE_PATH" \
    --open-threshold="$OPEN_THRESHOLD" \
    --closed-threshold="$CLOSED_THRESHOLD" \
    --min-closed-frames="$MIN_CLOSED_FRAMES" \
    --reward-value="$REWARD_VALUE"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$SHARED_WS/fabian/wandb}"
mkdir -p "$WANDB_DIR"

echo "=== Stage 3: training critic (num_qs=$NUM_QS) on combined reward (alpha_intrinsic=$ALPHA_INTRINSIC, alpha_goal=$ALPHA_GOAL) ==="
uv run scripts/train_qc_critic.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --data.root="$DATASET_ROOT" \
    --qc-cache-path="$QC_CACHE_PATH" \
    --horizon-length="$HORIZON_LENGTH" \
    --proprio-dim="$PROPRIO_DIM" \
    --action-dim="$ACTION_DIM" \
    --batch-size="$BATCH_SIZE" \
    --num-train-steps="$NUM_TRAIN_STEPS" \
    --checkpoint-dir="$CRITIC_CHECKPOINT_DIR" \
    --alpha-intrinsic="$ALPHA_INTRINSIC" \
    --alpha-goal="$ALPHA_GOAL" \
    --num-qs="$NUM_QS"

echo "Done. Subgoal-augmented critic checkpoint at $CRITIC_CHECKPOINT_DIR/final"
