#!/bin/bash
#SBATCH --job-name=prepare_libero_dataset
#SBATCH --partition=all
# No --gres: this is a CPU-only, no-GPU data-prep step (parquet/image
# decode + re-record via LeRobotDataset.create()/add_frame()/save_episode(),
# see scripts/convert_libero_dataset.py's module docstring for why this
# rebuild step exists at all -- the official lerobot v2.1->v3.0 converter
# doesn't support this dataset's actual v2.0 source format).
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# ONE-TIME step: converts physical-intelligence/libero into a real local
# v3.0 dataset on shared storage. Run this once before submitting any
# train_pi05_libero_jepa_sweep.slurm jobs -- they all read from
# $DATASET_ROOT set here. Safe to re-run: it checks whether $DATASET_ROOT
# already looks complete and skips if so (delete it first to force a
# from-scratch reconversion; see convert_libero_dataset.py's docstring on
# why this isn't resumable in place).

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/libero}"
REPO_ID="${REPO_ID:-physical-intelligence/libero}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

cd "$REPO_DIR"
mkdir -p slurm/logs

if [ -f "$DATASET_ROOT/meta/info.json" ] && [ -f "$DATASET_ROOT/meta/stats.json" ]; then
    echo "Dataset already present at $DATASET_ROOT (meta/info.json + meta/stats.json found) -- skipping."
    echo "Delete $DATASET_ROOT first if you want to force a from-scratch reconversion."
    exit 0
fi

uv run scripts/convert_libero_dataset.py \
    --repo-id="$REPO_ID" \
    --out-root="$DATASET_ROOT" \
    --image-writer-threads="${SLURM_CPUS_PER_TASK:-16}"

echo "Done. Libero v3.0 dataset ready at $DATASET_ROOT"
