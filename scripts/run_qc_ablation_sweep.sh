#!/bin/bash
# Runs the same critic-scoring ablation battery we ran on the first subgoal
# critic (see .qc_eval_run_logs/chain_4settings_newcritic.sh and
# chain_ablations_round2.sh on this machine, not checked into the repo) against
# a NEW checkpoint + critic pair -- generalized so it isn't hardcoded to one
# checkpoint like those one-off scripts were.
#
# Stage 1 picks a score direction (argmax vs argmin over the critic's raw
# score -- see src/qc/actor.py's best_of_n_action_batch) by running both
# plain, then layers critic-disagreement and actor-disagreement penalties on
# top of whichever direction wins.
# Stage 2 (only run if --full-sweep is passed) refines around the actor-
# disagreement result: does it also rescue the losing direction, does
# stacking critic-disagreement on top help further, and a small weight/
# sample-count sweep around actor-disagreement=1.0.
#
# Usage:
#   ./scripts/run_qc_ablation_sweep.sh <checkpoint_dir> <critic_checkpoint_path> <run_prefix> [num_trials] [num_qs] [--full-sweep]
#
# Example (once stopgrad_v3's full-JEPA extraction is downloaded locally):
#   ./scripts/run_qc_ablation_sweep.sh \
#       ~/openpi_eval_checkpoints_full_finetune_full_jepa/stopgrad_v3/29999 \
#       ~/openpi_qc_checkpoint/final \
#       stopgrad_v3_subgoal 20 5 --full-sweep

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CKPT_DIR="${1:?Usage: $0 <checkpoint_dir> <critic_checkpoint_path> <run_prefix> [num_trials] [num_qs] [--full-sweep]}"
CRITIC_PATH="${2:?Usage: $0 <checkpoint_dir> <critic_checkpoint_path> <run_prefix> [num_trials] [num_qs] [--full-sweep]}"
RUN_PREFIX="${3:?Usage: $0 <checkpoint_dir> <critic_checkpoint_path> <run_prefix> [num_trials] [num_qs] [--full-sweep]}"
NUM_TRIALS="${4:-20}"
NUM_QS="${5:-5}"
FULL_SWEEP=""
for arg in "$@"; do
    if [ "$arg" = "--full-sweep" ]; then
        FULL_SWEEP=1
    fi
done

LOGDIR="$HOME/.qc_eval_run_logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/ablation_sweep_${RUN_PREFIX}.log"
DECISION_FILE="$LOGDIR/${RUN_PREFIX}_direction_decision.txt"

export LC_ALL=C   # system locale is de_DE (comma decimal separator) -- avoid awk/numeric corruption

run_setting () {
    local name="$1"
    local uncertainty_penalty="$2"
    local actor_disagreement_penalty="$3"
    local maximize="$4"          # "true" or ""
    local critic_weight="${5:-1.0}"
    local num_samples="${6:-16}"

    echo "=== [$(date)] Starting setting: $name (unc_pen=$uncertainty_penalty, actor_pen=$actor_disagreement_penalty, maximize=$maximize, critic_weight=$critic_weight, num_samples=$num_samples) ===" | tee -a "$LOG"

    NUM_QS="$NUM_QS" \
    UNCERTAINTY_PENALTY="$uncertainty_penalty" \
    ACTOR_DISAGREEMENT_PENALTY="$actor_disagreement_penalty" \
    CRITIC_WEIGHT="$critic_weight" \
    MAXIMIZE_SCORE="$maximize" \
    SELECTION_MODE=score \
    NUM_SAMPLES="$num_samples" \
    RESULTS_FILE="libero_qc_${RUN_PREFIX}_${name}_all_suites_eval_results.txt" \
    ./scripts/eval_libero_qc_all_suites.sh \
        "$CKPT_DIR" "$CRITIC_PATH" "$NUM_TRIALS" "qc_${RUN_PREFIX}_${name}" \
        >> "$LOG" 2>&1

    echo "=== [$(date)] Finished setting: $name ===" | tee -a "$LOG"
}

avg_success () {
    # Averages every "Total success rate: X.XX" value in the given results file.
    local file="$1"
    grep -oE 'Total success rate: [0-9]+\.[0-9]+' "$file" | grep -oE '[0-9]+\.[0-9]+' \
        | awk '{sum+=$1; n++} END {if (n>0) printf "%.4f", sum/n; else print "0"}'
}

echo "=== [$(date)] Stage 1: direction pick (argmax vs argmin), prefix=$RUN_PREFIX ===" | tee -a "$LOG"
run_setting "argmax" 0.0 0.0 "true"
run_setting "argmin" 0.0 0.0 ""

ARGMAX_AVG="$(avg_success "libero_qc_${RUN_PREFIX}_argmax_all_suites_eval_results.txt")"
ARGMIN_AVG="$(avg_success "libero_qc_${RUN_PREFIX}_argmin_all_suites_eval_results.txt")"

WINNER="argmax"
WINNER_MAXIMIZE="true"
if awk -v a="$ARGMIN_AVG" -v b="$ARGMAX_AVG" 'BEGIN { exit !(a > b) }'; then
    WINNER="argmin"
    WINNER_MAXIMIZE=""
fi
echo "$WINNER" > "$DECISION_FILE"
echo "=== [$(date)] Direction decision: argmax avg=$ARGMAX_AVG, argmin avg=$ARGMIN_AVG -> WINNER=$WINNER ===" | tee -a "$LOG"

run_setting "${WINNER}_criticdisagree" 1.0 0.0 "$WINNER_MAXIMIZE"
run_setting "${WINNER}_actordisagree"  0.0 1.0 "$WINNER_MAXIMIZE"

echo "=== [$(date)] Stage 1 (4 settings) done ===" | tee -a "$LOG"

if [ -n "$FULL_SWEEP" ]; then
    LOSER="argmin"
    LOSER_MAXIMIZE=""
    if [ "$WINNER" = "argmin" ]; then
        LOSER="argmax"
        LOSER_MAXIMIZE="true"
    fi

    echo "=== [$(date)] Stage 2: refinement around ${WINNER}_actordisagree ===" | tee -a "$LOG"

    # 1. Does actor-disagreement rescue the losing direction too, or is it
    #    specific to the winner?
    run_setting "${LOSER}_actordisagree" 0.0 1.0 "$LOSER_MAXIMIZE" 1.0 16

    # 2. Actor-disagree + critic-disagree stacked on top of the winning base.
    run_setting "${WINNER}_actordisagree_criticdisagree" 1.0 1.0 "$WINNER_MAXIMIZE" 1.0 16

    # 3-4. Actor-disagreement weight sweep around 1.0.
    run_setting "${WINNER}_actordisagree_w0.5" 0.0 0.5 "$WINNER_MAXIMIZE" 1.0 16
    run_setting "${WINNER}_actordisagree_w2.0" 0.0 2.0 "$WINNER_MAXIMIZE" 1.0 16

    # 5. More candidates per step for the winning config.
    run_setting "${WINNER}_actordisagree_n32" 0.0 1.0 "$WINNER_MAXIMIZE" 1.0 32

    echo "=== [$(date)] ALL ABLATIONS DONE (prefix=$RUN_PREFIX, base direction=$WINNER) ===" | tee -a "$LOG"
else
    echo "=== [$(date)] Skipping stage 2 refinement sweep (pass --full-sweep to run it) ===" | tee -a "$LOG"
fi
