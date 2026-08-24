#!/bin/bash
#SBATCH --job-name=extract_libero_full_finetune_plain_control
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # MUST match the training run's --fsdp-devices=4 -- restoring a
                                          # checkpoint saved under N-way FSDP sharding needs that many real
                                          # JAX devices actually visible (same requirement as every other
                                          # extract_libero_full_finetune_*.slurm script)
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=256G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Extracts a servable (plain Pi0Config-shaped), flat/device-count-agnostic
# checkpoint for train_pi05_libero_full_finetune_plain_control.slurm's run
# (plain scripts/train.py, no JEPA/QC code at all) so it can be rsynced
# locally and compared against beta0.0/beta0.1/stopgrad/betadecay via
# eval_libero_all_steps_all_suites.sh. This run has no JEPA wrapper to strip
# (unlike the beta0.0/beta0.1/stopgrad/betadecay runs, which go through
# OpenPIWithJEPA even at beta_jepa=0.0), but STILL needs this step: the raw
# checkpoint was saved under 4-way FSDP sharding (--fsdp-devices=4, matching
# --gres above) and can't be restored on a single local GPU without it.

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
OUT_DIR="${OUT_DIR:-$SHARED_WS/fabian/eval_checkpoints_full_finetune/plain_control}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
EXP_NAME="${EXP_NAME:-libero_full_finetune_plain_control}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"        # must match the training run's --fsdp-devices

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$OUT_DIR"

uv run scripts/extract_base_model_checkpoint.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --out-dir="$OUT_DIR" \
    --fsdp-devices="$FSDP_DEVICES"

echo "Done. Extracted checkpoints under $OUT_DIR/<step>/ -- rsync locally:"
echo "  rsync -avz <cluster>:$OUT_DIR ~/openpi_eval_checkpoints/eval_checkpoints_full_finetune/plain_control"
