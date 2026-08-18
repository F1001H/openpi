#!/bin/bash
#SBATCH --job-name=pi05_libero_full_finetune_beta_decay
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:3                # same footprint as train_pi05_libero_full_finetune.slurm --
                                          # the schedule only changes a scalar multiplier, not memory usage
#SBATCH --cpus-per-task=56               # 112 CPUs / 8 GPUs per node -- proportional 4-GPU share
#SBATCH --mem=512G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -euo pipefail

# Full (non-LoRA) fine-tune with beta_jepa COSINE-DECAYED from BETA_JEPA_INIT
# down to BETA_JEPA_FINAL over BETA_JEPA_DECAY_STEPS, instead of held
# constant (config.py's TrainConfig.beta_jepa_final/beta_jepa_decay_steps,
# implemented in src/jepa/train_step_transitions.py's train_step). Motivated
# by the same step-level eval breakdown as the stopgrad ablation
# (libero_all_steps_all_suites_eval_results*.txt): at constant beta_jepa=0.1,
# full fine-tuning tracks or beats the beta_jepa=0.0 baseline through step
# ~15000-20000, then falls behind as BC converges and JEPA's gradient starts
# purely competing for capacity rather than adding anything BC hasn't
# already learned. Decaying beta tries to keep the early-training benefit
# (shared representation shaping while there's still structure to learn)
# without paying the late-training cost, in a single training run -- no
# second stage, no extra compute versus the constant-beta run it's compared
# against.
#
# Default: 0.5 -> 0.0 over the full 30k-step run (BETA_JEPA_DECAY_STEPS
# defaults to config.num_train_steps when unset). Starts higher than the
# sweep's winning constant value (0.1) since the whole point is that a
# larger early push is affordable if it's fully gone by the time BC
# convergence would otherwise let it start hurting -- if this starts too
# strong and the early steps look like the beta>=0.25 collapse pattern from
# the LoRA sweep, lower BETA_JEPA_INIT.
#
# Compare this checkpoint's step-level eval curve against beta0.0 (no JEPA),
# constant beta0.1 (the run that lost), and (if run) the stopgrad ablation
# (train_pi05_libero_full_finetune_stopgrad.slurm) -- these are two
# independent, combinable hypotheses (decay addresses "which steps get
# gradient", stopgrad addresses "which weights get gradient"); worth trying
# together (--jepa-stopgrad-vision + this schedule) if either helps alone.

REPO_DIR="${REPO_DIR:-$HOME/openpi}"
SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
DATASET_ROOT="${DATASET_ROOT:-$SHARED_WS/fabian/libero}"
JEPA_PT_CKPT="${JEPA_PT_CKPT:-$SHARED_WS/fabian/vjepa2-ac-vitg.pt}"
JEPA_NPZ_CKPT="${JEPA_NPZ_CKPT:-$SHARED_WS/fabian/jepa_predictor/vjepa2_ac_converted.npz}"

export HF_HOME="${HF_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vast/workspaces/VLA_Reinforcement/openpi_cache}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$SHARED_WS/fabian/checkpoints}"
export WANDB_DIR="${WANDB_DIR:-$SHARED_WS/fabian/wandb}"

CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
ALPHA_BC="${ALPHA_BC:-1.0}"
BETA_JEPA_INIT="${BETA_JEPA_INIT:-0.5}"
BETA_JEPA_FINAL="${BETA_JEPA_FINAL:-0.0}"
BETA_JEPA_DECAY_STEPS="${BETA_JEPA_DECAY_STEPS:-}"   # empty -> defaults to num_train_steps (see config.py)
EXP_NAME="${EXP_NAME:-libero_jepa_full_finetune_betadecay_${BETA_JEPA_INIT}to${BETA_JEPA_FINAL}}"
FSDP_DEVICES="${FSDP_DEVICES:-3}"        # matches --gres above

cd "$REPO_DIR"
mkdir -p slurm/logs "$CHECKPOINT_BASE_DIR" "$WANDB_DIR" "$(dirname "$JEPA_NPZ_CKPT")"

if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "ERROR: dataset not found at $DATASET_ROOT -- run slurm/prepare_libero_dataset.slurm first." >&2
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

DECAY_STEPS_ARGS=()
if [ -n "$BETA_JEPA_DECAY_STEPS" ]; then
    DECAY_STEPS_ARGS=(--beta-jepa-decay-steps="$BETA_JEPA_DECAY_STEPS")
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_end_to_end.py \
    "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    --resume \
    --data.root="$DATASET_ROOT" \
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR" \
    --fsdp-devices="$FSDP_DEVICES" \
    --alpha-bc="$ALPHA_BC" \
    --beta-jepa="$BETA_JEPA_INIT" \
    --beta-jepa-final="$BETA_JEPA_FINAL" \
    "${DECAY_STEPS_ARGS[@]}" \
    --jepa-predictor-checkpoint "$JEPA_NPZ_CKPT"

echo "Done. Checkpoints under $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/"
