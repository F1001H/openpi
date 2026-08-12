#!/bin/bash
#SBATCH --job-name=pi05_libero_full_finetune_baseline
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # same memory pressure as train_pi05_libero_full_finetune.slurm
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=512G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Full (non-LoRA) fine-tune BASELINE -- same everything as
# train_pi05_libero_full_finetune.slurm except beta_jepa=0.0 (no JEPA
# co-training at all). Mathematically equivalent to plain BC:
# loss = alpha_bc*loss_bc + beta_jepa*loss_jepa, so beta_jepa=0 means zero
# gradient contribution from the JEPA term into every param, shared backbone
# included -- exactly how the LoRA sweep's own beta0.0 point (76% success)
# already served as its "no JEPA" baseline, not a separate plain-BC script.
# Kept as its own named script (rather than just BETA_JEPA=0.0 sbatch
# train_pi05_libero_full_finetune.slurm) to match the kobo convention of one
# dedicated script per condition (train_pi0_kobo_cube_bc_only.slurm etc.),
# for discoverability.
#
# Exists to give the non-LoRA (full fine-tune) regime its own baseline to
# compare against train_pi05_libero_full_finetune.slurm's beta_jepa=0.1 run
# -- same reasoning as the LoRA sweep, now checking whether the LoRA-vs-full
# question and the beta_jepa-weighting question interact.

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
BETA_JEPA="${BETA_JEPA:-0.0}"             # the only real difference from train_pi05_libero_full_finetune.slurm
EXP_NAME="${EXP_NAME:-libero_jepa_full_finetune_beta${BETA_JEPA}}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"

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
