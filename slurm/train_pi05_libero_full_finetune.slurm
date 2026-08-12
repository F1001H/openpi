#!/bin/bash
#SBATCH --job-name=pi05_libero_full_finetune
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # full (non-LoRA) fine-tune + JEPA predictor -- same memory pressure
                                          # that made train_pi0_kobo_cube_full_finetune.slurm need multi-GPU FSDP
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=512G                       # matches train_pi0_kobo_cube_full_finetune.slurm's note: full fine-tune's
                                          # optimizer state alone needs ~41GB staged in host memory at checkpoint save
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Full (non-LoRA) fine-tune counterpart to the beta_jepa sweep (which used
# pi05_libero_low_mem, LoRA) -- isolates whether LoRA's limited trainable-
# parameter budget is leaving performance on the table at the sweep's
# winning weighting (beta_jepa=0.1: 78% success at 30k steps, best AND most
# stable of the sweep -- see full sweep+eval results). pi05_libero (config.py)
# is the exact full-finetune counterpart to pi05_libero_low_mem, same as
# pi0_kobo_cube is to pi0_kobo_cube_low_mem -- no new config needed, just a
# different CONFIG_NAME plus the extra GPUs/FSDP full fine-tuning needs.
#
# Uses pi05_libero's OWN established batch_size=256/lr_schedule (its
# upstream-tuned full-finetune hyperparameters) rather than forcing the
# LoRA sweep's batch_size=32 -- these were tuned for full fine-tuning
# specifically; forcing LoRA's settings would risk an apples-to-oranges
# comparison in the other direction. If you want a stricter LoRA-vs-full
# comparison holding batch_size/lr fixed instead, override BATCH_SIZE below.

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/libero}"
JEPA_PT_CKPT="${JEPA_PT_CKPT:-$SHARED_WS/fabian/vjepa2-ac-vitg.pt}"
JEPA_NPZ_CKPT="${JEPA_NPZ_CKPT:-$SHARED_WS/fabian/jepa_predictor/vjepa2_ac_converted.npz}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
export WANDB_DIR="${WANDB_DIR:-$SHARED_WS/fabian/wandb}"

CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
ALPHA_BC="${ALPHA_BC:-1.0}"
BETA_JEPA="${BETA_JEPA:-0.1}"
EXP_NAME="${EXP_NAME:-libero_jepa_full_finetune_beta${BETA_JEPA}}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"        # matches --gres above -- without this, config.fsdp_devices defaults to
                                          # 1 and the extra allocated GPUs sit idle instead of sharding the model

cd "$REPO_DIR"
mkdir -p slurm/logs "$CHECKPOINT_BASE_DIR" "$WANDB_DIR" "$(dirname "$JEPA_NPZ_CKPT")"

if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "ERROR: dataset not found at $DATASET_ROOT -- run slurm/prepare_libero_dataset.slurm first." >&2
    exit 1
fi
if [ ! -f "$DATASET_ROOT/meta/stats.json" ]; then
    echo "ERROR: $DATASET_ROOT/meta/stats.json missing -- norm stats can't be loaded." >&2
    exit 1
fi

if [ ! -f "$JEPA_NPZ_CKPT" ]; then
    if [ ! -f "$JEPA_PT_CKPT" ]; then
        echo "ERROR: $JEPA_PT_CKPT not found -- copy vjepa2-ac-vitg.pt to the cluster first." >&2
        exit 1
    fi
    echo "Converting V-JEPA2-AC checkpoint (one-time)..."
    uv run python -m jepa.convert_checkpoint --ckpt "$JEPA_PT_CKPT" --dry-run
    uv run python -m jepa.convert_checkpoint --ckpt "$JEPA_PT_CKPT" --save-path "$JEPA_NPZ_CKPT"
fi

export WANDB_MODE="${WANDB_MODE:-offline}"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_end_to_end.py \
    "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --resume \
    --data.root="$DATASET_ROOT" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --fsdp-devices="$FSDP_DEVICES" \
    --alpha-bc="$ALPHA_BC" \
    --beta-jepa="$BETA_JEPA" \
    --jepa-predictor-checkpoint "$JEPA_NPZ_CKPT"

echo "Done. Checkpoints under $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/"
