#!/bin/bash
#SBATCH --job-name=train_libero_qc_critic
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Trains the Q-chunking critic (scripts/train_qc_critic.py) against the
# reward cache slurm/label_libero_qc_rewards.slurm produced. Small MLP over
# a cached (embedding, proprio, action, reward) dataset -- no VLA/JEPA model
# loaded here, so this is much lighter/faster than the labeling pass (the
# original kobo critic converged within ~10k steps of a similarly-sized
# critic).

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/libero}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_low_mem}"
EXP_NAME="${EXP_NAME:-libero_jepa_sweep_alpha1.0_beta0.1}"
QC_CACHE_PATH="${QC_CACHE_PATH:-$SHARED_WS/fabian/qc_cache_libero_beta0.1.npz}"
CRITIC_CHECKPOINT_DIR="${CRITIC_CHECKPOINT_DIR:-$SHARED_WS/fabian/checkpoints_qc/libero_beta0.1}"

HORIZON_LENGTH="${HORIZON_LENGTH:-5}"
PROPRIO_DIM="${PROPRIO_DIM:-8}"
ACTION_DIM="${ACTION_DIM:-7}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$CRITIC_CHECKPOINT_DIR"

if [ ! -f "$QC_CACHE_PATH" ]; then
    echo "ERROR: reward cache not found at $QC_CACHE_PATH -- run slurm/label_libero_qc_rewards.slurm first." >&2
    exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$SHARED_WS/fabian/wandb}"
mkdir -p "$WANDB_DIR"

uv run scripts/train_qc_critic.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --data.root="$DATASET_ROOT" \
    --qc-cache-path="$QC_CACHE_PATH" \
    --horizon-length="$HORIZON_LENGTH" \
    --proprio-dim="$PROPRIO_DIM" \
    --action-dim="$ACTION_DIM" \
    --batch-size="$BATCH_SIZE" \
    --num-train-steps="$NUM_TRAIN_STEPS" \
    --checkpoint-dir="$CRITIC_CHECKPOINT_DIR"

echo "Done. Critic checkpoint at $CRITIC_CHECKPOINT_DIR/final"
