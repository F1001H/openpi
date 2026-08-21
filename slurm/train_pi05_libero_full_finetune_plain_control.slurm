#!/bin/bash
#SBATCH --job-name=pi05_libero_full_finetune_plain_control
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # same footprint as train_pi05_libero_full_finetune*.slurm
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=512G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# TRUE unmodified-pipeline control, unlike train_pi05_libero_full_finetune_
# baseline.slurm's beta_jepa=0.0 run. That run (and every other LIBERO
# full-finetune run in this repo) goes through scripts/train_end_to_end.py,
# wired to the from-scratch JepaTransitionDataLoader (src/utils/data_loader.
# py) instead of openpi.training.data_loader.create_data_loader -- an
# "independent reimplementation" per CLAUDE.md, never cross-checked against
# the standard pipeline, and we already found one real bug in that custom
# pipeline this session (ema_params/attn_mask checkpoint corruption, see
# openpi/training/checkpoints.py's _prune_to_structure). beta_jepa=0.0 zeroes
# the JEPA LOSS's gradient contribution, but says nothing about whether the
# surrounding data loading/batching/shuffling is actually identical to the
# reference pipeline plain scripts/train.py uses -- the same pipeline that
# (presumably) produced examples/libero/README.md's official 96.85% avg
# number, and that every other openpi/lerobot user reproducing pi0.5 on
# LIBERO is also using.
#
# This run calls scripts/train.py DIRECTLY -- zero JEPA/QC code involved,
# same config (pi05_libero), same 30k steps, same data.root. If this closes
# most of the gap to beta0.0's 90.6% (or better yet the official 96.85%),
# the JepaTransitionDataLoader pipeline itself is the culprit, separate from
# anything about JEPA co-training's loss/gradient effects. If it lands in
# the same ~87-91% range as our existing beta0.0 run, the pipeline isn't the
# issue and the gap is elsewhere (eval harness -- see slurm/extract_libero_
# full_finetune_stopgrad_checkpoint.slurm's sibling official-checkpoint-eval
# experiment -- or a genuine hyperparameter/data difference).

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/libero}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
export WANDB_DIR="${WANDB_DIR:-$SHARED_WS/fabian/wandb}"

CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
EXP_NAME="${EXP_NAME:-libero_full_finetune_plain_control}"
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
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --fsdp-devices="$FSDP_DEVICES"

echo "Done. Checkpoints under $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/"
