#!/bin/bash
#SBATCH --job-name=pi0_kobo_cube_bc_only
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

# Plain BC-only training run (NO JEPA co-training at all -- scripts/train.py,
# not scripts/train_end_to_end.py) on the SAME dataset/config as
# full_lora_test, to directly test whether JEPA co-training is responsible
# for full_lora_test's failure to track the cube's actual position (see
# memory: project_qchunking_posttraining.md, "MAJOR finding 2026-07-31").
# Deliberately mirrors train_pi0_kobo_cube.slurm's structure/paths so the
# ONLY real difference between this run and full_lora_test is JEPA
# co-training on vs off -- same dataset, same config, same norm_stats fix.

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/kobo_cube}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

# Checkpoints (and wandb's local offline run data) go to the shared 10TB
# workspace, not the 200GB personal home-dir quota -- Fabian was running low
# on home-dir space. checkpoint_base_dir is a plain TrainConfig field
# (config.py:658, default "./checkpoints" i.e. under REPO_DIR), overridden
# via --checkpoint-base-dir below same as any other CLI field.
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
export WANDB_DIR="${WANDB_DIR:-$SHARED_WS/fabian/wandb}"

CONFIG_NAME="${CONFIG_NAME:-pi0_kobo_cube_low_mem}"
# Deliberately NOT timestamped, same convention as train_pi0_kobo_cube.slurm --
# plain `sbatch` reruns/resumes THIS run; pass EXP_NAME=<something-else>
# sbatch ... to start a genuinely separate run.
EXP_NAME="${EXP_NAME:-bc_only_test1}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$CHECKPOINT_BASE_DIR" "$WANDB_DIR"

if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "ERROR: dataset not found at $DATASET_ROOT (expected $DATASET_ROOT/meta/info.json etc.)" >&2
    exit 1
fi
if [ ! -f "$DATASET_ROOT/meta/stats.json" ]; then
    echo "ERROR: $DATASET_ROOT/meta/stats.json missing -- norm stats can't be loaded." >&2
    exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"

# scripts/train.py is the plain openpi BC entrypoint -- no JEPA predictor,
# no OpenPIWithJEPA wrapper, no jepa-predictor-checkpoint arg. Same
# num_train_steps=30_000/batch_size=32 as the config's own defaults, same
# --resume-ability convention as the JEPA slurm script.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --resume \
    --data.root="$DATASET_ROOT" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR"

echo "Done. Checkpoints under $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/"
