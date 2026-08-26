#!/bin/bash
#SBATCH --job-name=extract_libero_full_finetune_data_fraction
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # MUST match the training run's --fsdp-devices=4 -- restoring a
                                          # checkpoint saved under N-way FSDP sharding needs that many real
                                          # JAX devices actually visible (same requirement as every other
                                          # extract_libero_full_finetune_*.slurm script)
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=256G
#SBATCH --exclude=cn02,cn17              # cn17: confirmed broken. cn02: intermittently can't reach its
                                          # storage. Override on the sbatch command line if these get fixed.
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Extracts a servable (plain Pi0Config-shaped), flat/device-count-agnostic
# checkpoint for one data-fraction sweep point (train_pi05_libero_full_
# finetune_data_fraction.slurm, EPISODE_FRACTION=0.5/0.25/0.1) -- plain
# scripts/train.py, no JEPA/QC code, same as plain_control's extraction.
# EXP_NAME and OUT_DIR are BOTH derived from EPISODE_FRACTION by default so
# each sweep point lands in its own distinct output dir automatically --
# the betadecay extraction script's shared-OUT_DIR-by-default footgun (all
# 3 variants defaulting to the same path, clobbering each other) doesn't
# repeat here.
#
# Uses scripts/extract_plain_checkpoint.py, NOT extract_base_model_
# checkpoint.py -- the latter assumes every checkpoint is OpenPIWithJEPA-
# wrapped (true for beta0.0/beta0.1/stopgrad/betadecay, NOT true here) and
# crashes on restore. See extract_libero_full_finetune_plain_control_
# checkpoint.slurm's identical note -- same underlying bug, same fix,
# confirmed via a real crash extracting plain_control this way.
#
# Usage: EPISODE_FRACTION=0.1 sbatch slurm/extract_libero_full_finetune_data_fraction_checkpoint.slurm

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
EPISODE_FRACTION="${EPISODE_FRACTION:?Usage: EPISODE_FRACTION=0.1 sbatch $0 (e.g. 0.1, 0.25, 0.5 -- must match the value the training job used)}"
EXP_NAME="${EXP_NAME:-libero_full_finetune_data_fraction_${EPISODE_FRACTION}}"
OUT_DIR="${OUT_DIR:-$SHARED_WS/fabian/eval_checkpoints_full_finetune/data_fraction_${EPISODE_FRACTION}}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"        # must match the training run's --fsdp-devices

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$OUT_DIR"

uv run scripts/extract_plain_checkpoint.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --out-dir="$OUT_DIR" \
    --fsdp-devices="$FSDP_DEVICES"

echo "Done. Extracted checkpoints under $OUT_DIR/<step>/ -- rsync locally:"
echo "  rsync -avz <cluster>:$OUT_DIR ~/openpi_eval_checkpoints/eval_checkpoints_full_finetune/data_fraction_${EPISODE_FRACTION}"
