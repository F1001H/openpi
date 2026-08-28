#!/bin/bash
#SBATCH --job-name=extract_libero_full_finetune_beta0.0_v2
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # MUST match the training run's --fsdp-devices=4 -- restoring a
                                          # checkpoint saved under N-way FSDP sharding needs that many real
                                          # JAX devices actually visible (same requirement as every other
                                          # extract_libero_full_finetune_*.slurm script)
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=256G
#SBATCH --exclude=cn02,cn17              # cn17: confirmed broken. cn02: intermittently can't reach its
                                          # storage. Override on the sbatch command line if these get fixed.
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Extracts the beta0.0_v2 verification run (constant beta_jepa=0.0, i.e. zero
# JEPA gradient -- isolates whether JepaTransitionDataLoader's now-fixed
# batch-mixing/reshuffling bugs alone close the gap to plain_control, with no
# JEPA loss confound at all) -- see train_pi05_libero_full_finetune_baseline.
# slurm, resubmitted as EXP_NAME=libero_jepa_full_finetune_beta0.0_v2.
#
# Deliberately a DEDICATED script rather than reusing extract_libero_full_
# finetune_checkpoints.slurm's BETA_JEPA_GRID loop: that script's OUT_DIR is
# "$OUT_ROOT/beta0.0" for the beta=0.0 grid point, which already holds the
# ORIGINAL (pre-any-fix) beta0.0 reference checkpoint from the very start of
# this investigation -- reusing it here without care would clobber that
# reference data. This script's own OUT_DIR default is a separate,
# non-colliding path instead.

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
OUT_DIR="${OUT_DIR:-$SHARED_WS/fabian/eval_checkpoints_full_finetune/beta0.0_v2}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
EXP_NAME="${EXP_NAME:-libero_jepa_full_finetune_beta0.0_v2}"
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
echo "  rsync -avz <cluster>:$OUT_DIR ~/openpi_eval_checkpoints/eval_checkpoints_full_finetune/beta0.0_v2"
