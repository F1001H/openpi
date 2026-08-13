#!/bin/bash
#SBATCH --job-name=extract_libero_full_finetune_checkpoints
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # MUST match the training runs' --fsdp-devices=4 -- restoring a
                                          # checkpoint saved under N-way FSDP sharding needs that many real
                                          # JAX devices actually visible, not just a matching --fsdp-devices
                                          # flag (confirmed: 1 GPU here raised orbax's
                                          # "sharding passed to deserialization should be ... Got None")
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=256G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Extracts servable (plain Pi0Config-shaped) checkpoints for the two full
# (non-LoRA) fine-tune runs -- train_pi05_libero_full_finetune.slurm
# (beta_jepa=0.1, the LoRA sweep's winner) and
# train_pi05_libero_full_finetune_baseline.slurm (beta_jepa=0.0, its
# baseline) -- so they can be rsynced locally and evaluated (see
# scripts/eval_libero_all_suites.sh for the 4-suite comparison). Mirrors
# slurm/extract_libero_sweep_checkpoints.slurm exactly, just against
# pi05_libero (full fine-tune) instead of pi05_libero_low_mem (LoRA), and
# only 2 runs instead of 5. scripts/extract_base_model_checkpoint.py
# extracts EVERY surviving checkpoint step per run by default (not just the
# latest), one out_dir/<step>/ subdirectory each -- for 2 runs x ~6 steps
# each, expect a few minutes total (same per-extraction cost as the LoRA
# sweep: restore ~14GB, slice out base_model, write ~7GB, roughly 15-30s
# each -- full fine-tune checkpoints are the same base_model size as LoRA
# ones once extracted, only the training-time optimizer state differs).

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
OUT_ROOT="${OUT_ROOT:-$SHARED_WS/fabian/eval_checkpoints_full_finetune}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
BETA_JEPA_GRID="${BETA_JEPA_GRID:-0.0 0.1}"
EXP_PREFIX="${EXP_PREFIX:-libero_jepa_full_finetune}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"        # must match train_pi05_libero_full_finetune*.slurm's --fsdp-devices

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$OUT_ROOT"

for beta in $BETA_JEPA_GRID; do
    exp_name="${EXP_PREFIX}_beta${beta}"
    out_dir="$OUT_ROOT/beta${beta}"
    echo "Extracting $exp_name -> $out_dir"
    uv run scripts/extract_base_model_checkpoint.py "$CONFIG_NAME" \
        --exp-name="$exp_name" \
        --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
        --out-dir="$out_dir" \
        --fsdp-devices="$FSDP_DEVICES"
done

echo "Done. Extracted checkpoints under $OUT_ROOT/beta<value>/ -- rsync this whole directory locally:"
echo "  rsync -avz <cluster>:$OUT_ROOT ~/openpi_eval_checkpoints_full_finetune"
