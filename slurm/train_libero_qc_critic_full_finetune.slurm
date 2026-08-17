#!/bin/bash
#SBATCH --job-name=train_libero_qc_critic_full_finetune
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Full-finetune counterpart to slurm/train_libero_qc_critic.slurm: trains the
# Q-chunking critic (scripts/train_qc_critic.py) against the reward cache
# slurm/label_libero_qc_rewards_full_finetune.slurm produced
# ($SHARED_WS/fabian/qc_cache_libero_full_finetune_beta0.1.npz, from the
# FULLY TRAINED full-finetune beta_jepa=0.1 checkpoint at step 29999 -- see
# that script's comment on why beta0.0 isn't usable here: its jepa_predictor
# never received gradient, so it has nothing meaningful to label rewards
# with). Same lightweight MLP-over-cached-embeddings setup as the LoRA
# version -- no VLA/JEPA model loaded here, 1 GPU is plenty.
#
# CONFIG_NAME defaults to pi05_libero (full fine-tune, matches the checkpoint
# the cache was labeled from) rather than the LoRA sweep's pi05_libero_low_mem
# -- train_qc_critic.py only uses config.data/config.assets_dirs here (for
# dataset root resolution), not the model itself, but keeping it aligned with
# the source checkpoint avoids any assets-dir mismatch.
#
# Once this finishes, the resulting critic + the full-finetune base_model
# checkpoint (~/openpi_eval_checkpoints_full_finetune/beta0.1/29999, or the
# cluster path CHECKPOINT_BASE_DIR/pi05_libero/libero_jepa_full_finetune_beta0.1/29999)
# are what scripts/serve_qc_policy.py needs for the offline best-of-N eval,
# mirroring what was already validated end-to-end on the LoRA sweep's critic.

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/libero}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
EXP_NAME="${EXP_NAME:-libero_jepa_full_finetune_critic_beta0.1}"
QC_CACHE_PATH="${QC_CACHE_PATH:-$SHARED_WS/fabian/qc_cache_libero_full_finetune_beta0.1.npz}"
CRITIC_CHECKPOINT_DIR="${CRITIC_CHECKPOINT_DIR:-$SHARED_WS/fabian/checkpoints_qc/libero_full_finetune_beta0.1}"

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
    echo "ERROR: reward cache not found at $QC_CACHE_PATH -- run slurm/label_libero_qc_rewards_full_finetune.slurm first." >&2
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
