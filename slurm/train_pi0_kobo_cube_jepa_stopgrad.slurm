#!/bin/bash
#SBATCH --job-name=pi0_kobo_cube_jepa_stopgrad
#SBATCH --partition=all
# --account removed: cm74paso is Fabian's login username, not necessarily a
# valid SLURM account/project name -- omitting it lets SLURM use the default
# association for the user. If sbatch complains about a missing/required
# account, run `sacctmgr show associations user=$USER` to find the real name.
#SBATCH --gres=gpu:h100:1                # cn01's Gres=gpu:h100:8 confirms the GRES name is "h100"
#SBATCH --cpus-per-task=14               # 112 CPUs / 8 GPUs per node (cn01) -- proportional 1-GPU share
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# JEPA co-training WITH the shared-vision-backbone gradient path cut
# (--jepa-stopgrad-vision, config.py's TrainConfig.jepa_stopgrad_vision,
# default False elsewhere including full_lora_test). Mechanistic test of
# whether JEPA co-training degrading BC's spatial precision (see memory:
# project_qchunking_posttraining.md, "MAJOR finding 2026-07-31" + follow-up)
# is specifically caused by JEPA's loss backpropagating into the SAME
# PaliGemma/SigLIP weights BC's embed_prefix depends on. This variant still
# trains jepa_predictor/vision_proj/action_proj/state_proj normally (so it
# still produces a usable intrinsic-reward signal) but stop-gradients the
# vision features at the point they enter the JEPA path, so JEPA's loss can
# no longer touch the vision backbone at all.
#
# Compare this checkpoint's cube-tracking behavior against BOTH full_lora_test
# (joint gradients, doesn't track the cube) and bc_only_test1 (no JEPA at
# all, see train_pi0_kobo_cube_bc_only.slurm) once all three are done:
#   - stopgrad tracks the cube ~as well as bc_only_test1 -> confirms the
#     shared-backbone-gradient-interference mechanism specifically.
#   - stopgrad still fails like full_lora_test -> the interference isn't
#     via the shared vision backbone; look elsewhere (or it's really just
#     the data-diversity/reach-limit explanation and JEPA was a red herring).

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/kobo_cube}"
JEPA_PT_CKPT="${JEPA_PT_CKPT:-$SHARED_WS/fabian/vjepa2-ac-vitg.pt}"
JEPA_NPZ_CKPT="${JEPA_NPZ_CKPT:-$SHARED_WS/fabian/jepa_predictor/vjepa2_ac_converted.npz}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

# Shared workspace for checkpoints/wandb, not Fabian's home-dir quota -- same
# reasoning as train_pi0_kobo_cube_bc_only.slurm.
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
export WANDB_DIR="${WANDB_DIR:-$SHARED_WS/fabian/wandb}"

CONFIG_NAME="${CONFIG_NAME:-pi0_kobo_cube_low_mem}"
EXP_NAME="${EXP_NAME:-jepa_stopgrad_test1}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$CHECKPOINT_BASE_DIR" "$WANDB_DIR" "$(dirname "$JEPA_NPZ_CKPT")"

if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "ERROR: dataset not found at $DATASET_ROOT (expected $DATASET_ROOT/meta/info.json etc.)" >&2
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
    --jepa-stopgrad-vision \
    --jepa-predictor-checkpoint "$JEPA_NPZ_CKPT"

echo "Done. Checkpoints under $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/"
