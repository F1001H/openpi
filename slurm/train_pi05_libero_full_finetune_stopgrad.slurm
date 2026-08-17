#!/bin/bash
#SBATCH --job-name=pi05_libero_full_finetune_stopgrad
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # same footprint as train_pi05_libero_full_finetune.slurm --
                                          # stopgrad only changes where a gradient is cut, not memory usage
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=512G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Full (non-LoRA) fine-tune counterpart to train_pi0_kobo_cube_jepa_stopgrad.slurm's
# ablation, at the LoRA sweep's winning beta_jepa=0.1. Tests the leading
# hypothesis for why beta_jepa=0.1 REVERSES from beating the no-JEPA baseline
# (LoRA sweep, +2pp) to losing to it (full fine-tune, -2.2 to -2.3pp,
# replicated across seed=7 and seed=1234 -- see libero_all_steps_all_suites_
# eval_results*.txt): under full fine-tuning, JEPA's auxiliary gradient has a
# much bigger, more entangled shared PaliGemma/SigLIP backbone to compete
# with BC in than it does under LoRA's narrow adapter deltas. --jepa-stopgrad-
# vision (config.py's TrainConfig.jepa_stopgrad_vision) cuts JEPA's gradient
# into that shared backbone entirely while still training jepa_predictor/
# vision_proj/action_proj/state_proj normally, so the run still produces a
# usable intrinsic-reward signal for later QC labeling.
#
# Compare this checkpoint's step-level LIBERO eval curve against BOTH
# beta0.0 (no JEPA at all) and beta0.1 (joint gradients, the run that lost)
# once done:
#   - stopgrad recovers to ~beta0.0's level -> confirms the shared-backbone-
#     gradient-interference mechanism specifically; --jepa-stopgrad-vision
#     becomes the recommended default for full fine-tuning.
#   - stopgrad still underperforms like joint-gradient beta0.1 -> the
#     interference isn't via the shared vision backbone; look at the beta
#     decay schedule (train_pi05_libero_full_finetune_beta_decay.slurm)
#     instead, or the JEPA-predictor's own action/state projections.

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
EXP_NAME="${EXP_NAME:-libero_jepa_full_finetune_stopgrad_beta${BETA_JEPA}}"
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

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_end_to_end.py \
    "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --resume \
    --data.root="$DATASET_ROOT" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --fsdp-devices="$FSDP_DEVICES" \
    --alpha-bc="$ALPHA_BC" \
    --beta-jepa="$BETA_JEPA" \
    --jepa-stopgrad-vision \
    --jepa-predictor-checkpoint "$JEPA_NPZ_CKPT"

echo "Done. Checkpoints under $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/"
