#!/bin/bash
# Submits one train_pi05_libero_jepa_sweep.slurm job per beta_jepa value in
# BETA_JEPA_GRID (alpha_bc held fixed at ALPHA_BC) -- follow-up to the
# bc_only_test1/jepa_stopgrad_test1 ablations, isolating JEPA's relative
# loss weight now that pi05_libero_low_mem (Libero, unimanual) is available
# as a stand-in dataset while the real kobo robot is down.
#
# Run slurm/prepare_libero_dataset.slurm first (once) so $DATASET_ROOT
# already has the converted Libero dataset before these jobs start.
#
# Usage:
#   ./slurm/submit_libero_jepa_sweep.sh
# Override the grid or run length inline, e.g.:
#   NUM_TRAIN_STEPS=10000 ./slurm/submit_libero_jepa_sweep.sh
#   BETA_JEPA_GRID="0 0.5 1.0" ./slurm/submit_libero_jepa_sweep.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ALPHA_BC="${ALPHA_BC:-1.0}"
BETA_JEPA_GRID="${BETA_JEPA_GRID:-0.0 0.1 0.25 0.5 1.0}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
EXP_PREFIX="${EXP_PREFIX:-libero_jepa_sweep}"

for beta in $BETA_JEPA_GRID; do
    exp_name="${EXP_PREFIX}_alpha${ALPHA_BC}_beta${beta}"
    echo "Submitting $exp_name (alpha_bc=$ALPHA_BC, beta_jepa=$beta, num_train_steps=$NUM_TRAIN_STEPS)"
    ALPHA_BC="$ALPHA_BC" BETA_JEPA="$beta" NUM_TRAIN_STEPS="$NUM_TRAIN_STEPS" EXP_NAME="$exp_name" \
        sbatch slurm/train_pi05_libero_jepa_sweep.slurm
done

echo "Submitted $(echo $BETA_JEPA_GRID | wc -w) jobs. Track with: squeue -u \$USER"
