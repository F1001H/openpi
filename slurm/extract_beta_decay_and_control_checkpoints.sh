#!/bin/bash
# Submits all 5 extraction jobs (plain_control, the 3 beta-decay variants,
# stopgrad_v2) in one go, each with its correct EXP_NAME/OUT_DIR override --
# see slurm/extract_libero_full_finetune_*.slurm's own docs for why each
# needs a distinct OUT_DIR (the beta-decay script's default OUT_DIR is
# shared across all three variants; reusing it without an override would
# clobber one variant's extracted checkpoint with another's).
#
# Run this FROM THE CLUSTER CHECKOUT (it calls sbatch directly), and only
# after confirming each training job has actually reached a saved
# checkpoint post-cn02-exclusion -- submitting against a run with no
# checkpoints yet will just fail inside the job.
#
# Usage:
#   ./slurm/extract_beta_decay_and_control_checkpoints.sh
#   EXCLUDE_NODES=cn02,cn05 ./slurm/extract_beta_decay_and_control_checkpoints.sh
#   EXCLUDE_NODES= ./slurm/extract_beta_decay_and_control_checkpoints.sh   # no exclusion

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

EXCLUDE_NODES="${EXCLUDE_NODES:-cn02}"   # cn02 confirmed broken as of this session; override/clear once fixed
EXCLUDE_ARGS=()
if [ -n "$EXCLUDE_NODES" ]; then
    EXCLUDE_ARGS=(--exclude="$EXCLUDE_NODES")
fi

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
OUT_ROOT="$SHARED_WS/fabian/eval_checkpoints_full_finetune"

echo "=== Submitting extraction jobs (excluding: ${EXCLUDE_NODES:-none}) ==="

sbatch "${EXCLUDE_ARGS[@]}" \
    slurm/extract_libero_full_finetune_plain_control_checkpoint.slurm

EXP_NAME=libero_jepa_full_finetune_betadecay_0.5to0.0_v2 \
OUT_DIR="$OUT_ROOT/betadecay_v2" \
    sbatch "${EXCLUDE_ARGS[@]}" \
    slurm/extract_libero_full_finetune_betadecay_checkpoint.slurm

EXP_NAME=libero_jepa_full_finetune_betadecay_0.1to0.0 \
OUT_DIR="$OUT_ROOT/betadecay_conservative" \
    sbatch "${EXCLUDE_ARGS[@]}" \
    slurm/extract_libero_full_finetune_betadecay_checkpoint.slurm

EXP_NAME=libero_jepa_full_finetune_betadecay_0.5to0.05 \
OUT_DIR="$OUT_ROOT/betadecay_floor" \
    sbatch "${EXCLUDE_ARGS[@]}" \
    slurm/extract_libero_full_finetune_betadecay_checkpoint.slurm

EXP_NAME=libero_jepa_full_finetune_stopgrad_beta0.1_v2 \
    sbatch "${EXCLUDE_ARGS[@]}" \
    slurm/extract_libero_full_finetune_stopgrad_checkpoint.slurm

echo "=== Done. Check with: squeue -u \$USER -o '%.10i %.9P %.40j %.8T %.10M %.6D %R' ==="
