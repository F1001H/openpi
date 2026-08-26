#!/bin/bash
#SBATCH --job-name=label_libero_qc_rewards_full_finetune
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # MUST match the training run's --fsdp-devices=4 -- restoring an
                                          # N-way FSDP-sharded checkpoint needs that many real JAX devices
                                          # actually visible (see slurm/extract_libero_full_finetune_checkpoints.slurm's
                                          # identical note -- confirmed via a real orbax error there)
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=256G                       # same host-RAM budget as the LoRA version (label_libero_qc_rewards.slurm)
                                          # -- the OOM there was from DataLoader workers on host RAM, not GPU
                                          # count, so this doesn't need to scale with --gres
#SBATCH --exclude=cn02,cn17              # cn17: confirmed broken. cn02: intermittently can't reach its
                                          # storage. Override on the sbatch command line if these get fixed.
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Same reward + embedding + candidate-action labeling pass (scripts/
# qc_label_rewards.py) as slurm/label_libero_qc_rewards.slurm, but against
# the FULL (non-LoRA) fine-tune "trained model" run
# (train_pi05_libero_full_finetune.slurm, beta_jepa=0.1) instead of the LoRA
# sweep winner -- meant to run in parallel with/independent of the LIBERO
# sim eval jobs, on its own GPU allocation.
#
# Defaults to beta_jepa=0.1 (not the beta_jepa=0.0 baseline): with
# beta_jepa=0.0 the jepa_predictor receives exactly zero gradient during
# training (loss = alpha_bc*BC + beta_jepa*JEPA, so the JEPA term
# contributes nothing to any param when beta_jepa=0), meaning that
# checkpoint's jepa_predictor never moved from its V-JEPA2-AC initialization
# -- a much weaker/less meaningful basis for the intrinsic-reward signal
# this labeling pass computes than the beta_jepa=0.1 run, whose predictor
# was actually refined during co-training.
#
# SLOW: budget on the order of half a day (the LoRA version's real run took
# ~7.5-8hrs at these same conservative --num-workers/--reward-batch-size
# settings, notably faster than the naive ~271k-transitions/0.17s-per-frame
# extrapolation -- apparently more GPU-bound than CPU/IO-bound in practice).

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/libero}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
EXP_NAME="${EXP_NAME:-libero_jepa_full_finetune_beta0.1}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-29999}"
OUTPUT_PATH="${OUTPUT_PATH:-$SHARED_WS/fabian/qc_cache_libero_full_finetune_beta0.1.npz}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"        # must match train_pi05_libero_full_finetune.slurm's --fsdp-devices

HORIZON_LENGTH="${HORIZON_LENGTH:-5}"
ACTION_DIM="${ACTION_DIM:-7}"              # Libero's native action dim (vs kobo's 8)
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
# Same conservative defaults as label_libero_qc_rewards.slurm -- that
# script's original num_workers=14/--reward-batch-size=64 OOM'd host memory
# (5 oom_kill events); these values are already proven to work.
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"

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
    --fsdp-devices="$FSDP_DEVICES" \
    --output-path="$OUTPUT_PATH" \
    --horizon-length="$HORIZON_LENGTH" \
    --action-dim="$ACTION_DIM" \
    --num-candidates="$NUM_CANDIDATES" \
    --reward-batch-size="$REWARD_BATCH_SIZE" \
    --num-workers="$NUM_WORKERS"

echo "Done. QC reward cache at $OUTPUT_PATH"
