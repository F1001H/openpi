#!/bin/bash
#SBATCH --job-name=pi05_libero_jepa_sweep
#SBATCH --partition=all
# --account intentionally omitted -- see slurm/train_pi0_kobo_cube.slurm's
# comment: the login username isn't a valid SLURM account, and omitting it
# lets SLURM use the default association.
#SBATCH --gres=gpu:h100:1                # LoRA config (pi05_libero_low_mem) -- same 1-GPU footprint as the kobo LoRA runs
#SBATCH --cpus-per-task=14               # 112 CPUs / 8 GPUs per node -- proportional 1-GPU share
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Single run of the alpha_bc/beta_jepa loss-weighting sweep on Libero
# (unimanual stand-in while the real kobo robot is down -- see this
# session's earlier pipeline-validation work). Mirrors
# train_pi0_kobo_cube_jepa_stopgrad.slurm's structure/conventions, but:
#   - pi05_libero_low_mem instead of pi0_kobo_cube_low_mem (Libero dataset,
#     LoRA, JEPA co-training on).
#   - ALPHA_BC/BETA_JEPA are swept via env vars rather than fixed -- see
#     slurm/submit_libero_jepa_sweep.sh, which submits one job per
#     (alpha_bc, beta_jepa) combination with distinct EXP_NAMEs.
# Do NOT run this before slurm/prepare_libero_dataset.slurm has completed at
# least once -- it needs $DATASET_ROOT to already contain the converted
# v3.0 Libero dataset.

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/libero}"
JEPA_PT_CKPT="${JEPA_PT_CKPT:-$SHARED_WS/fabian/vjepa2-ac-vitg.pt}"
# Dataset-independent (same ViT-g V-JEPA2-AC weights regardless of which
# robot/sim dataset co-trains against them) -- reuses whatever the kobo runs
# already converted, no need to reconvert per dataset.
JEPA_NPZ_CKPT="${JEPA_NPZ_CKPT:-$SHARED_WS/fabian/jepa_predictor/vjepa2_ac_converted.npz}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
export WANDB_DIR="${WANDB_DIR:-$SHARED_WS/fabian/wandb}"

CONFIG_NAME="${CONFIG_NAME:-pi05_libero_low_mem}"
# alpha_bc/beta_jepa are TrainConfig fields (config.py) read by
# src/jepa/train_step_transitions.py's train_step; defaults there are
# alpha_bc=1.0, beta_jepa=0.5 -- both overridden here so the sweep launcher
# can vary them per job.
ALPHA_BC="${ALPHA_BC:-1.0}"
BETA_JEPA="${BETA_JEPA:-0.5}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
# EXP_NAME encodes the sweep point so distinct combinations don't collide
# under the same checkpoint dir -- --resume below only continues THIS exact
# EXP_NAME if resubmitted with the same value.
EXP_NAME="${EXP_NAME:-libero_jepa_sweep_beta${BETA_JEPA}}"

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

echo "Sweep point: EXP_NAME=$EXP_NAME ALPHA_BC=$ALPHA_BC BETA_JEPA=$BETA_JEPA"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_end_to_end.py \
    "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --resume \
    --data.root="$DATASET_ROOT" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --num-train-steps="$NUM_TRAIN_STEPS" \
    --alpha-bc="$ALPHA_BC" \
    --beta-jepa="$BETA_JEPA" \
    --jepa-predictor-checkpoint "$JEPA_NPZ_CKPT"

echo "Done. Checkpoints under $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/"
