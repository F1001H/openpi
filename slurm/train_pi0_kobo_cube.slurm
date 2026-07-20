#!/bin/bash
#SBATCH --job-name=pi0_kobo_cube
#SBATCH --partition=all
# --account removed: cm74paso is Fabian's login username, not necessarily a
# valid SLURM account/project name -- omitting it lets SLURM use the default
# association for the user. If sbatch complains about a missing/required
# account, run `sacctmgr show associations user=$USER` to find the real name.
#SBATCH --gres=gpu:h100:1                # cn01's Gres=gpu:h100:8 confirms the GRES name is "h100"
#SBATCH --cpus-per-task=14               # 112 CPUs / 8 GPUs per node (cn01) -- proportional 1-GPU share
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# --- 0. Fixed paths -- adjust if your clone/dataset live elsewhere on the cluster --------- #
REPO_DIR="${REPO_DIR:-$HOME/openpi}"
# NOTE: kobo_cube (dataset) and vjepa2-ac-vitg.pt (11GB pretrained checkpoint)
# are exactly the kind of shared project assets your coworker's shared-10TB-
# workspace policy is meant to cover -- defaulting these into the shared
# workspace rather than each person's 200GB personal one. This subdirectory
# layout (dataset/vjepa2 under VLA_Reinforcement/) is a guess, not a
# team-confirmed convention -- confirm/adjust with the team, and if someone's
# already put these there, just point at the real paths.
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/kobo_cube}"
JEPA_PT_CKPT="${JEPA_PT_CKPT:-$SHARED_WS/fabian/vjepa2-ac-vitg.pt}"
JEPA_NPZ_CKPT="${JEPA_NPZ_CKPT:-$SHARED_WS/fabian/jepa_predictor/vjepa2_ac_converted.npz}"

# Shared team workspace (10TB) vs. personal workspace (200GB per person) -- per
# team convention, cached pretrained models/datasets go in the shared
# workspace so we don't each keep our own copy. Two SEPARATE cache
# mechanisms need pointing there, not just one:
#   - HF_HOME: huggingface_hub/datasets caches -- this is what LeRobotDataset
#     actually uses under the hood (see the ~/.cache/huggingface/datasets
#     parquet-conversion cache we hit locally), so it's what your coworker
#     was referring to.
#   - OPENPI_DATA_HOME: openpi's OWN cache for its gs:// checkpoint downloads
#     (e.g. weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/
#     pi05_base/params") -- defaults to ~/.cache/openpi, see
#     src/openpi/shared/download.py). This is exactly the kind of "pretrained
#     model" duplication the shared-workspace policy is meant to avoid, but
#     setting HF_HOME alone does NOT cover it.
export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

CONFIG_NAME="pi0_kobo_cube"
# Deliberately NOT timestamped: --resume below only continues the same run if
# checkpoints/$CONFIG_NAME/$EXP_NAME/ matches the earlier submission. Plain
# `sbatch slurm/train_pi0_kobo_cube.slurm` reruns/resumes THIS run; pass
# EXP_NAME=<something-else> sbatch ... to start a genuinely separate run.
EXP_NAME="${EXP_NAME:-full_finetune_v1}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$(dirname "$JEPA_NPZ_CKPT")"

# --- 1. Sanity checks -- fail fast instead of burning GPU-hours on a bad config ----------- #
if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "ERROR: dataset not found at $DATASET_ROOT (expected $DATASET_ROOT/meta/info.json etc.)" >&2
    exit 1
fi
if [ ! -f "$DATASET_ROOT/meta/stats.json" ]; then
    # KoboDataConfig reads norm stats directly from the dataset's own
    # meta/stats.json (see DataConfigFactory._load_norm_stats in config.py) --
    # no separate `compute_norm_stats.py` run is needed for this data config,
    # AS LONG AS this file made it over with the rest of the dataset.
    echo "ERROR: $DATASET_ROOT/meta/stats.json missing -- norm stats can't be loaded." >&2
    exit 1
fi

# --- 2. Convert the JEPA predictor checkpoint once, cache the result --------------------- #
if [ ! -f "$JEPA_NPZ_CKPT" ]; then
    if [ ! -f "$JEPA_PT_CKPT" ]; then
        echo "ERROR: $JEPA_PT_CKPT not found -- copy vjepa2-ac-vitg.pt to the cluster first." >&2
        exit 1
    fi
    echo "Converting V-JEPA2-AC checkpoint (one-time)..."
    uv run python -m jepa.convert_checkpoint --ckpt "$JEPA_PT_CKPT" --dry-run
    uv run python -m jepa.convert_checkpoint --ckpt "$JEPA_PT_CKPT" --save-path "$JEPA_NPZ_CKPT"
fi

# --- 3. wandb -- set WANDB_API_KEY in your shell env or ~/.netrc before submitting, or ---- #
#        export WANDB_MODE=offline / pass --no-wandb-enabled below to disable it entirely.
export WANDB_MODE="${WANDB_MODE:-offline}"

# --- 4. Train ----------------------------------------------------------------------------- #
# pi0_kobo_cube is the FULL (non-LoRA) fine-tune. Locally, adding the ViT-g
# JEPA predictor on top OOM'd a 32GB card (needed ~58GB) -- pi0_kobo_cube_low_mem
# (LoRA) was used for all local debugging instead. If this OOMs here too,
# either drop --batch-size, add --fsdp-devices <n> to shard across more GPUs
# in this job, or fall back to pi0_kobo_cube_low_mem.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_end_to_end.py \
    "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --resume \
    --data.root="$DATASET_ROOT" \
    --jepa-predictor-checkpoint "$JEPA_NPZ_CKPT" \
    --num-train-steps=100 \

echo "Done. Checkpoints under checkpoints/$CONFIG_NAME/$EXP_NAME/"
