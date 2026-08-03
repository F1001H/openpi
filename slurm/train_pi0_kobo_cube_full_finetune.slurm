#!/bin/bash
#SBATCH --job-name=pi0_kobo_cube_full_finetune
#SBATCH --partition=all
# --account removed: cm74paso is Fabian's login username, not necessarily a
# valid SLURM account/project name -- omitting it lets SLURM use the default
# association for the user. If sbatch complains about a missing/required
# account, run `sacctmgr show associations user=$USER` to find the real name.
#SBATCH --gres=gpu:h100:4                # cn01's Gres=gpu:h100:8 confirms the GRES name is "h100"
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node (cn01) -- proportional 1-GPU share
# 128G was NOT enough -- got a SLURM oom_kill (host RAM, not GPU VRAM) right
# at step 1000's checkpoint save. Full (non-LoRA) fine-tune's optimizer state
# alone needs ~41GB staged in host memory during the save (see the
# "Transferring arrays to host memory" / train_state write in the crash log),
# on top of the running process's own footprint and per-GPU staging buffers
# (4 GPUs each contributing their own). cn01-class nodes have ~2TB RAM total
# across 8 GPUs, so 512G leaves generous headroom without meaningfully
# competing for the node's capacity.
#SBATCH --mem=512G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Full (non-LoRA) fine-tune of pi0_kobo_cube, WITH JEPA co-training kept on
# (same as full_lora_test) -- isolates a DIFFERENT hypothesis than the
# bc_only_test1 / jepa_stopgrad_test1 runs: maybe the issue isn't JEPA at
# all, but LoRA's limited trainable-parameter budget not being expressive
# enough to learn precise cube-position coupling. Same dataset, same
# norm_stats fix, same JEPA co-training setup as full_lora_test -- the ONLY
# real difference is pi0_kobo_cube (full fine-tune) vs pi0_kobo_cube_low_mem
# (LoRA).
#
# Memory note (carried over from train_pi0_kobo_cube.slurm): full fine-tune +
# the ViT-g JEPA predictor OOM'd a 32GB local card (needed ~58GB) -- should
# fit on an 80GB H100 with headroom, but if it OOMs here too, drop
# --batch-size or add --fsdp-devices <n> to shard across more GPUs in this
# job (request more GPUs via --gres above first).

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/kobo_cube}"
JEPA_PT_CKPT="${JEPA_PT_CKPT:-$SHARED_WS/fabian/vjepa2-ac-vitg.pt}"
JEPA_NPZ_CKPT="${JEPA_NPZ_CKPT:-$SHARED_WS/fabian/jepa_predictor/vjepa2_ac_converted.npz}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

# Shared workspace for checkpoints/wandb, not Fabian's home-dir quota -- same
# reasoning as the other new slurm scripts in this batch.
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
export WANDB_DIR="${WANDB_DIR:-$SHARED_WS/fabian/wandb}"

# NOTE: full fine-tune config, NOT _low_mem -- this is the whole point of
# this script. Do not override to _low_mem; use train_pi0_kobo_cube.slurm
# for that.
CONFIG_NAME="${CONFIG_NAME:-pi0_kobo_cube}"
EXP_NAME="${EXP_NAME:-full_finetune_jepa_test1}"

cd "$REPO_DIR"
mkdir -p slurm/logs "$CHECKPOINT_BASE_DIR" "$WANDB_DIR" "$(dirname "$JEPA_NPZ_CKPT")"

if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "ERROR: dataset not found at $DATASET_ROOT (expected $DATASET_ROOT/meta/info.json etc.)" >&2
    exit 1
fi
if [ ! -f "$DATASET_ROOT/meta/stats.json" ]; then
    echo "ERROR: $DATASET_ROOT/meta/stats.json missing -- norm stats can't be loaded." >&2
    exit 1
fi

if [ ! -f "$JEPA_NPZ_CKPT" ]; then
    if [ ! -f "$JEPA_PT_CKPT" ]; then
        echo "ERROR: $JEPA_PT_CKPT not found -- copy vjepa2-ac-vitg.pt to the cluster first." >&2
        exit 1
    fi
    echo "Converting V-JEPA2-AC checkpoint (one-time)..."
    uv run python -m jepa.convert_checkpoint --ckpt "$JEPA_PT_CKPT" --dry-run
    uv run python -m jepa.convert_checkpoint --ckpt "$JEPA_PT_CKPT" --save-path "$JEPA_NPZ_CKPT"
fi

export WANDB_MODE="${WANDB_MODE:-offline}"

# FSDP_DEVICES matches the #SBATCH --gres=gpu:h100:4 request above -- without
# --fsdp-devices, config.fsdp_devices defaults to 1 and the mesh only uses a
# single device, leaving the second allocated GPU idle instead of actually
# sharding the model to get the memory headroom the 2-GPU request is for.
FSDP_DEVICES="${FSDP_DEVICES:-4}"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_end_to_end.py \
    "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --resume \
    --data.root="$DATASET_ROOT" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --fsdp-devices="$FSDP_DEVICES" \
    --jepa-predictor-checkpoint "$JEPA_NPZ_CKPT"

echo "Done. Checkpoints under $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/"
