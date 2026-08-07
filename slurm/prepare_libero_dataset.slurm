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

# Anonymous/unauthenticated requests (esp. to HF's "Xet" storage backend's
# .../xet-read-token endpoint) hit a 429 from this cluster's IP range while
# downloading physical-intelligence/libero (a heavily-shared dataset).
# huggingface_hub's get_token() checks the HF_TOKEN env var BEFORE the
# $HF_HOME/token file -- deliberately relying on that here instead of
# `huggingface-cli login`, since HF_HOME above is the SHARED team workspace
# and `login` would write the token file there, exposing your personal
# token to everyone else with access to it. Instead, add this to your own
# ~/.bashrc (private, in your home dir, never touches shared storage):
#   export HF_TOKEN=hf_...
# (token from https://huggingface.co/settings/tokens -- "Read" scope is
# enough). sbatch inherits your shell's environment by default, so it flows
# into every job automatically with no per-job passing needed. This only
# warns, doesn't fail -- a token isn't strictly required, just makes the 429
# far less likely.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: no HF_TOKEN env var set -- downloads will be unauthenticated and more" >&2
    echo "         likely to hit a 429 rate limit. Add 'export HF_TOKEN=hf_...' to your own" >&2
    echo "         ~/.bashrc (get a token at https://huggingface.co/settings/tokens) to fix" >&2
    echo "         this for good -- do NOT run 'huggingface-cli login' with HF_HOME set to" >&2
    echo "         the shared workspace, that would write your token where others can read it." >&2
fi

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
