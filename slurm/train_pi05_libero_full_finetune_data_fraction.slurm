#!/bin/bash
#SBATCH --job-name=pi05_libero_full_finetune_data_fraction
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # same footprint as train_pi05_libero_full_finetune_plain_control.slurm
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=512G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# BC-only (plain scripts/train.py, no JEPA/QC code -- same as plain_control)
# reference curve for the sample-efficiency sweep: trains on EPISODE_FRACTION
# of each task's demos instead of all of them, via DataConfig.episode_fraction
# (src/openpi/training/config.py) -- stratified BY TASK (src/openpi/training/
# data_loader.py's _select_episodes_by_fraction), so every task keeps at
# least 1 episode at any fraction instead of risking a task being dropped
# entirely at low fractions.
#
# This is needed regardless of which JEPA/QC variant (constant beta, decay
# schedule, stopgrad) eventually proves "not worse than BC" at full data --
# the actual paper claim is SAMPLE EFFICIENCY (does the augmented method's
# advantage over BC grow as available demos shrink), and that requires a
# BC-only baseline at each data fraction to compare against. Safe to run
# now, in parallel with the still-training JEPA variants, since it doesn't
# depend on which of them wins.
#
# Norm stats are NOT recomputed per fraction (computed once from the full
# dataset, held fixed) -- see DataConfig.episode_fraction's docstring for
# why: keeps this sweep an apples-to-apples "same normalization, less data"
# comparison rather than also varying normalization statistics between
# conditions.
#
# Usage: EPISODE_FRACTION=0.1 sbatch slurm/train_pi05_libero_full_finetune_data_fraction.slurm
#   (0.25, 0.5 are the other planned sweep points -- 1.0 is plain_control itself)

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/libero}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
export WANDB_DIR="${WANDB_DIR:-$SHARED_WS/fabian/wandb}"

CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
EPISODE_FRACTION="${EPISODE_FRACTION:?Usage: EPISODE_FRACTION=0.1 sbatch $0 (e.g. 0.1, 0.25, 0.5)}"
EXP_NAME="${EXP_NAME:-libero_full_finetune_data_fraction_${EPISODE_FRACTION}}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"        # matches --gres above

cd "$REPO_DIR"
mkdir -p slurm/logs "$CHECKPOINT_BASE_DIR" "$WANDB_DIR"

if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "ERROR: dataset not found at $DATASET_ROOT -- run slurm/prepare_libero_dataset.slurm first." >&2
    exit 1
fi
if [ ! -f "$DATASET_ROOT/meta/stats.json" ]; then
    echo "ERROR: $DATASET_ROOT/meta/stats.json missing -- norm stats can't be loaded." >&2
    exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --resume \
    --data.root="$DATASET_ROOT" \
    --data.episode-fraction="$EPISODE_FRACTION" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --fsdp-devices="$FSDP_DEVICES"

echo "Done. Checkpoints under $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/"
