#!/bin/bash
#SBATCH --job-name=pi05_libero_full_finetune_beta_decay_conservative
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # same footprint as train_pi05_libero_full_finetune.slurm --
                                          # the schedule only changes a scalar multiplier, not memory usage
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=512G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Second beta-decay variant, alongside train_pi05_libero_full_finetune_beta_
# decay.slurm's default (0.5 -> 0.0). That run starts ABOVE the sweep's
# winning constant value (0.1) on the theory that a stronger early push is
# affordable if fully gone before BC convergence would let it start hurting.
# This one instead starts AT the already-validated-safe constant value and
# decays it to 0 -- isolates whether decay helps AT ALL from a starting
# point already known not to cause the LoRA sweep's beta>=0.25 collapse
# pattern, separate from the "does a stronger-than-0.1 early push help more"
# question the 0.5-start run is testing. If this ties or beats the constant-
# 0.1 baseline, decay's benefit is coming from removing late-training
# competition, not from the extra early push; if it doesn't, the 0.5-start
# run's result (if positive) would instead say the extra push mattered more
# than the late-training removal.
#
# Compare against: beta0.0 (no JEPA), constant beta0.1 (lost late), the
# 0.5->0.0 decay run, and stopgrad -- all four are cheap to compare via
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
BETA_JEPA_INIT="${BETA_JEPA_INIT:-0.1}"
BETA_JEPA_FINAL="${BETA_JEPA_FINAL:-0.0}"
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
