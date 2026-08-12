#!/bin/bash
#SBATCH --job-name=label_libero_qc_rewards
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1                # LoRA config, same 1-GPU footprint as the sweep training runs
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# One-time reward + embedding + candidate-action labeling pass (scripts/
# qc_label_rewards.py) over the FULL converted Libero dataset, using the
# sweep's winning checkpoint (beta_jepa=0.1 -- see the full sweep+eval
# results: it was the only run to both beat plain BC (78% vs 76% at
# beta=0.0) AND train stably the whole way, unlike beta>=0.25 which
# collapsed late in training). Needs the FULL (unextracted) JEPA-wrapped
# training checkpoint -- NOT scripts/extract_base_model_checkpoint.py's
# output, which strips out exactly the jepa_predictor/extract_vision_latents
# this script needs.
#
# SLOW: this is a full forward pass (BC actor sampling + JEPA prediction
# error) over ~271k valid transitions. A local 4-episode/~1100-frame smoke
# test ran at roughly 0.17s/frame -- extrapolated to the full dataset, budget
# on the order of half a day to a day, not minutes. Runs unattended; check
# progress via the tqdm-logged batch counter in this job's log.

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/libero}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_low_mem}"
EXP_NAME="${EXP_NAME:-libero_jepa_sweep_alpha1.0_beta0.1}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-29999}"
OUTPUT_PATH="${OUTPUT_PATH:-$SHARED_WS/fabian/qc_cache_libero_beta0.1.npz}"

HORIZON_LENGTH="${HORIZON_LENGTH:-5}"
ACTION_DIM="${ACTION_DIM:-7}"              # Libero's native action dim (vs kobo's 8)
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-64}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$(dirname "$OUTPUT_PATH")"

if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "ERROR: dataset not found at $DATASET_ROOT -- run slurm/prepare_libero_dataset.slurm first." >&2
    exit 1
fi
if [ ! -d "$CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/$CHECKPOINT_STEP" ]; then
    echo "ERROR: checkpoint step $CHECKPOINT_STEP not found under $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME" >&2
    exit 1
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/qc_label_rewards.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --data.root="$DATASET_ROOT" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --checkpoint-step="$CHECKPOINT_STEP" \
    --output-path="$OUTPUT_PATH" \
    --horizon-length="$HORIZON_LENGTH" \
    --action-dim="$ACTION_DIM" \
    --num-candidates="$NUM_CANDIDATES" \
    --reward-batch-size="$REWARD_BATCH_SIZE" \
    --num-workers="${SLURM_CPUS_PER_TASK:-14}"

echo "Done. QC reward cache at $OUTPUT_PATH"
