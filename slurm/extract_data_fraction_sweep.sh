#!/bin/bash
# Submits extraction for all 3 data-fraction sweep points (0.5, 0.25, 0.1)
# at once, each landing in its own OUT_DIR automatically (derived from
# EPISODE_FRACTION by extract_libero_full_finetune_data_fraction_checkpoint.
# slurm itself -- no manual OUT_DIR bookkeeping needed here, unlike the
# beta-decay extraction script).
#
# Only submit a given fraction once its training job has actually reached a
# saved checkpoint -- this doesn't check that itself.
#
# Run this FROM THE CLUSTER CHECKOUT (it calls sbatch directly).
#
# Usage:
#   ./slurm/extract_data_fraction_sweep.sh
#   ./slurm/extract_data_fraction_sweep.sh 0.25 0.1   # only specific fractions

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FRACTIONS=("$@")
if [ "${#FRACTIONS[@]}" -eq 0 ]; then
    FRACTIONS=(0.5 0.25 0.1)
fi

for frac in "${FRACTIONS[@]}"; do
    echo "=== Submitting extraction for EPISODE_FRACTION=$frac ==="
    EPISODE_FRACTION="$frac" sbatch slurm/extract_libero_full_finetune_data_fraction_checkpoint.slurm
done

echo "=== Done. Check with: squeue -u \$USER -o '%.10i %.9P %.40j %.8T %.10M %.6D %R' ==="
