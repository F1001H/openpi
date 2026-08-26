#!/bin/bash
# Submits QC reward-labeling (scripts/qc_label_rewards.py, via slurm/label_
# libero_qc_rewards_full_finetune.slurm) for BOTH stopgrad_v2 and
# betadecay_conservative -- our two best fixed-RoPE/fixed-dataloader
# checkpoints (92.0% / 91.9% avg), meant to run overnight in parallel (each
# gets its own 4-GPU allocation).
#
# stopgrad_v2 is the stronger candidate for THIS pipeline specifically --
# its beta_jepa stays constant throughout training (only the backbone-
# directed gradient is cut), so its jepa_predictor trained for the full 30k
# steps. betadecay_conservative's beta_jepa decays to 0.0 by the end, so
# its predictor's training effectively tapered off near the final
# checkpoint even though its BC policy performance held up fine -- worth
# comparing the resulting reward caches/critics once both land, since a
# weaker predictor could mean a noisier intrinsic-reward signal even if the
# base policy itself is comparable.
#
# Each run takes on the order of half a day per the template's own note.
#
# Run this FROM THE CLUSTER CHECKOUT (it calls sbatch directly).
#
# Usage:
#   ./slurm/label_stopgrad_and_betadecay_conservative.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"

EXP_NAME=libero_jepa_full_finetune_stopgrad_beta0.1_v2 \
OUTPUT_PATH="$SHARED_WS/fabian/qc_cache_libero_full_finetune_stopgrad_v2.npz" \
    sbatch slurm/label_libero_qc_rewards_full_finetune.slurm

EXP_NAME=libero_jepa_full_finetune_betadecay_0.1to0.0 \
OUTPUT_PATH="$SHARED_WS/fabian/qc_cache_libero_full_finetune_betadecay_conservative.npz" \
    sbatch slurm/label_libero_qc_rewards_full_finetune.slurm

echo "=== Done. Check with: squeue -u \$USER -o '%.10i %.9P %.40j %.8T %.10M %.6D %R' ==="
