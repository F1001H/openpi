#!/bin/bash
#SBATCH --job-name=pi05_libero_full_finetune_beta_decay_floor
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # same footprint as train_pi05_libero_full_finetune.slurm --
                                          # the schedule only changes a scalar multiplier, not memory usage
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=512G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Third beta-decay variant. train_pi05_libero_full_finetune_beta_decay.slurm
# decays 0.5 -> 0.0 (full removal by end of training); train_pi05_libero_
# full_finetune_beta_decay_conservative.slurm decays 0.1 -> 0.0 (same, from
# a safer start). Both fully zero out beta_jepa late, on the theory that
# JEPA's gradient purely competes for capacity once BC has converged. This
# run instead keeps the SAME aggressive start (0.5) but decays only down to
# a small non-zero floor (0.05) -- tests whether fully eliminating the JEPA
# signal late is actually necessary, or whether a small residual co-training
# pressure throughout is harmless/beneficial (e.g. as a regularizer) and the
# late-training degradation at constant beta_jepa=0.1 was specifically about
# the MAGNITUDE competing with BC, not JEPA gradient flowing at all. If this
# tracks the 0.5->0.0 run's step-level curve closely, the floor doesn't
# matter (decay-to-zero is fine, simpler); if it degrades similarly to
# constant beta0.1 late in training, some in fact needs to hit exactly zero.
#
# Compare against: beta0.0 (no JEPA), constant beta0.1 (lost late), the
# 0.5->0.0 and 0.1->0.0 decay runs, and stopgrad -- all cheap to compare via
# eval_libero_all_steps_all_suites.sh once extracted (see slurm/extract_
# libero_full_finetune_betadecay_checkpoint.slurm, override EXP_NAME to
# match this run's).

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
BETA_JEPA_INIT="${BETA_JEPA_INIT:-0.5}"
BETA_JEPA_FINAL="${BETA_JEPA_FINAL:-0.05}"
BETA_JEPA_DECAY_STEPS="${BETA_JEPA_DECAY_STEPS:-}"   # empty -> defaults to num_train_steps (see config.py)
EXP_NAME="${EXP_NAME:-libero_jepa_full_finetune_betadecay_${BETA_JEPA_INIT}to${BETA_JEPA_FINAL}}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"        # matches --gres above

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

DECAY_STEPS_ARGS=()
if [ -n "$BETA_JEPA_DECAY_STEPS" ]; then
    DECAY_STEPS_ARGS=(--beta-jepa-decay-steps="$BETA_JEPA_DECAY_STEPS")
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_end_to_end.py \
    "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --resume \
    --data.root="$DATASET_ROOT" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --fsdp-devices="$FSDP_DEVICES" \
    --alpha-bc="$ALPHA_BC" \
    --beta-jepa="$BETA_JEPA_INIT" \
    --beta-jepa-final="$BETA_JEPA_FINAL" \
    "${DECAY_STEPS_ARGS[@]}" \
    --jepa-predictor-checkpoint "$JEPA_NPZ_CKPT"

echo "Done. Checkpoints under $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/"
