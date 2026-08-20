#!/bin/bash
#SBATCH --job-name=extract_libero_full_finetune_betadecay
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # MUST match the training run's --fsdp-devices=4 -- restoring a
                                          # checkpoint saved under N-way FSDP sharding needs that many real
                                          # JAX devices actually visible (same requirement as
                                          # extract_libero_full_finetune_checkpoints.slurm)
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=256G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Extracts a servable (plain Pi0Config-shaped) checkpoint for the beta-decay
# co-training ablation (train_pi05_libero_full_finetune_beta_decay.slurm --
# beta_jepa cosine-decayed BETA_JEPA_INIT -> BETA_JEPA_FINAL over the run,
# instead of held constant) so it can be rsynced locally and compared against
# beta0.0/beta0.1/stopgrad via eval_libero_all_steps_all_suites.sh, same as
# those already are. Mirrors extract_libero_full_finetune_stopgrad_checkpoint.
# slurm exactly, just for this run.
#
# EXP_NAME must match whatever the training run actually used -- defaults to
# train_pi05_libero_full_finetune_beta_decay.slurm's own default
# (libero_jepa_full_finetune_betadecay_${BETA_JEPA_INIT}to${BETA_JEPA_FINAL},
# i.e. libero_jepa_full_finetune_betadecay_0.5to0.0 unless BETA_JEPA_INIT/
# BETA_JEPA_FINAL were overridden at training time) -- override EXP_NAME
# directly if so, rather than trying to reconstruct it from those two vars.

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
OUT_DIR="${OUT_DIR:-$SHARED_WS/fabian/eval_checkpoints_full_finetune/betadecay}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
EXP_NAME="${EXP_NAME:-libero_jepa_full_finetune_betadecay_0.5to0.0}"
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
echo "  rsync -avz <cluster>:$OUT_DIR ~/openpi_eval_checkpoints/eval_checkpoints_full_finetune/betadecay"
