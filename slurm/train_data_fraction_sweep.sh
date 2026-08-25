#!/bin/bash
# Submits all 3 data-fraction sweep points (0.5, 0.25, 0.1) at once. 1.0 is
# plain_control itself, not resubmitted here. Node exclusions (cn02, cn17)
# are baked into train_pi05_libero_full_finetune_data_fraction.slurm's own
# #SBATCH --exclude= directive -- nothing to pass here unless overriding.
#
# Run this FROM THE CLUSTER CHECKOUT (it calls sbatch directly).
#
# Usage:
#   ./slurm/train_data_fraction_sweep.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

for frac in 0.5 0.25 0.1; do
    echo "=== Submitting EPISODE_FRACTION=$frac ==="
    EPISODE_FRACTION="$frac" sbatch slurm/train_pi05_libero_full_finetune_data_fraction.slurm
done

echo "=== Done. Check with: squeue -u \$USER -o '%.10i %.9P %.40j %.8T %.10M %.6D %R' ==="
