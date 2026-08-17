#!/bin/bash
#SBATCH --job-name=extract_libero_full_finetune_full_jepa
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

# One-time consolidation step (scripts/extract_full_jepa_checkpoint.py) that
# re-saves the FULL OpenPIWithJEPA-wrapped full-finetune beta_jepa=0.1
# checkpoint (base_model + jepa_predictor + vision_proj + action_proj +
# state_proj + target_norm -- everything scripts/serve_qc_policy.py's
# critic-scored best-of-N eval needs, unlike extract_libero_full_finetune_
# checkpoints.slurm's base_model-only extraction) as a flat, device-count-
# agnostic checkpoint. This 4-GPU restore only has to happen ONCE per step;
# the output is then rsync-able and servable locally on 1 GPU via
# serve_qc_policy.py's --checkpoint-dir flag, same as the plain-BC eval
# checkpoints already are via serve_policy.py.
#
# Only targets beta_jepa=0.1 (not the beta0.0 baseline) -- same reasoning as
# label_libero_qc_rewards_full_finetune.slurm: at beta_jepa=0.0 the
# jepa_predictor never received gradient, so there's nothing meaningful for
# best-of-N to score candidate actions with.

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
OUT_DIR="${OUT_DIR:-$SHARED_WS/fabian/eval_checkpoints_full_finetune_full_jepa/beta0.1}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
EXP_NAME="${EXP_NAME:-libero_jepa_full_finetune_beta0.1}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"        # must match train_pi05_libero_full_finetune.slurm's --fsdp-devices

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$OUT_DIR"

uv run scripts/extract_full_jepa_checkpoint.py "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --out-dir="$OUT_DIR" \
    --fsdp-devices="$FSDP_DEVICES"

echo "Done. Extracted full-JEPA checkpoints under $OUT_DIR/<step>/ -- rsync locally:"
echo "  rsync -avz <cluster>:$OUT_DIR ~/openpi_eval_checkpoints_full_finetune_full_jepa"
echo "Then serve any step (1 GPU, fully local):"
echo "  uv run scripts/serve_qc_policy.py $CONFIG_NAME --exp-name=$EXP_NAME \\"
echo "      --checkpoint-dir=~/openpi_eval_checkpoints_full_finetune_full_jepa/<step> \\"
echo "      --critic-checkpoint-path=<critic/final> --proprio-dim=8 --action-dim=7 --horizon-length=5"
