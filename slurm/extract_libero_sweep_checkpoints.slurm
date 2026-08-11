#!/bin/bash
#SBATCH --job-name=extract_libero_sweep_checkpoints
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1                # needs a real JAX device for checkpoint restore/save; login nodes have none
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Extracts servable (plain Pi0Config-shaped) checkpoints for every point in
# the alpha_bc/beta_jepa sweep (see slurm/submit_libero_jepa_sweep.sh), so
# they can be rsynced locally and evaluated in the LIBERO sim (see
# scripts/eval_libero_sweep_local.sh). Each extraction is I/O-bound (restore
# ~14GB, slice out base_model, write ~7GB) and takes roughly 15-30s -- the
# whole loop should finish in a few minutes, well within a single short job.

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
OUT_ROOT="${OUT_ROOT:-$SHARED_WS/fabian/eval_checkpoints}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_low_mem}"
ALPHA_BC="${ALPHA_BC:-1.0}"
BETA_JEPA_GRID="${BETA_JEPA_GRID:-0.0 0.1 0.25 0.5 1.0}"
EXP_PREFIX="${EXP_PREFIX:-libero_jepa_sweep}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$OUT_ROOT"

for beta in $BETA_JEPA_GRID; do
    exp_name="${EXP_PREFIX}_alpha${ALPHA_BC}_beta${beta}"
    out_dir="$OUT_ROOT/beta${beta}"
    echo "Extracting $exp_name -> $out_dir"
    uv run scripts/extract_base_model_checkpoint.py "$CONFIG_NAME" \
        --exp-name="$exp_name" \
        --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
        --out-dir="$out_dir"
done

echo "Done. Extracted checkpoints under $OUT_ROOT/beta<value>/ -- rsync this whole directory locally:"
echo "  rsync -avz <cluster>:$OUT_ROOT ~/openpi_eval_checkpoints"
