#!/bin/bash
# Runs the LIBERO sim eval against a SINGLE checkpoint across all four
# official suites (libero_spatial, libero_object, libero_goal, libero_10),
# matching the standard comparison table format (see examples/libero/
# README.md's own reference numbers for pi05_libero). Plain-BC
# (scripts/serve_policy.py) -- same scope as eval_libero_sweep_local.sh, not
# the QC critic-augmented path.
#
# Usage:
#   ./scripts/eval_libero_all_suites.sh <checkpoint_dir> [num_trials_per_task] [name]
#
# <checkpoint_dir> must directly contain {params,assets} (i.e. already
# pointing at one extracted step -- see scripts/extract_base_model_checkpoint.py).
#
# Example:
#   ./scripts/eval_libero_all_suites.sh ~/openpi_eval_checkpoints/full_finetune_beta0.1/29999 20

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CKPT_DIR="${1:?Usage: $0 <checkpoint_dir> [num_trials_per_task] [name]}"
NUM_TRIALS="${2:-20}"
NAME="${3:-$(basename "$(dirname "$CKPT_DIR")")_$(basename "$CKPT_DIR")}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"   # full fine-tune's config, NOT pi05_libero_low_mem -- override if evaluating a LoRA checkpoint instead
PORT="${PORT:-8000}"
# 3600s (the old hardcoded value) was sized for NUM_TRIALS=20 (200 episodes/
# suite) -- confirmed too tight at NUM_TRIALS=50 (500 episodes/suite): a real
# libero_10 50-trial run hit it at 484/500 episodes (83.5% success, matching
# the already-known 20-trial number closely -- a timeout, not a crash).
# 7200s covers 500 episodes with real margin at libero_10's observed
# ~7.4s/episode plain-BC pace; override for even larger NUM_TRIALS.
SUITE_TIMEOUT="${SUITE_TIMEOUT:-7200}"

if [ ! -d "$CKPT_DIR/params" ]; then
    echo "ERROR: $CKPT_DIR has no params/ subdir -- point this at a single extracted checkpoint step." >&2
    exit 1
fi

LIBERO_VENV="examples/libero/.venv"
export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"
export LIBERO_CONFIG_PATH="$PWD/third_party/libero/.libero_config"

RESULTS_FILE="${RESULTS_FILE:-libero_all_suites_eval_results.txt}"
: > "$RESULTS_FILE"

SUITES=(libero_spatial libero_object libero_goal libero_10)

for suite in "${SUITES[@]}"; do
    run_name="${NAME}_${suite}"
    echo "=== Evaluating $run_name ($CKPT_DIR) ==="
    video_dir="libero_eval_videos/$run_name"
    mkdir -p "$video_dir"

    # --port must come BEFORE the policy:checkpoint subcommand token -- see
    # eval_libero_sweep_local.sh's identical note (tyro subcommand parsing
    # doesn't accept top-level Args fields after it).
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py --port="$PORT" policy:checkpoint \
        --policy.config="$CONFIG_NAME" --policy.dir="$CKPT_DIR" \
        > "$video_dir/server.log" 2>&1 &
    server_pid=$!

    server_up=false
    for _ in $(seq 1 120); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            break
        fi
        if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
            server_up=true
            break
        fi
        sleep 2
    done

    if [ "$server_up" != true ]; then
        echo "$run_name: SERVER FAILED TO START (see $video_dir/server.log)" | tee -a "$RESULTS_FILE"
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        continue
    fi

    set +e
    (
        source "$LIBERO_VENV/bin/activate"
        MUJOCO_GL=egl timeout "$SUITE_TIMEOUT" python examples/libero/main.py \
            --args.port "$PORT" \
            --args.task-suite-name "$suite" --args.num-trials-per-task "$NUM_TRIALS" \
            --args.video-out-path "$video_dir"
    ) > "$video_dir/eval.log" 2>&1
    eval_status=$?
    set -e

    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true

    if [ $eval_status -ne 0 ]; then
        echo "$run_name: EVAL FAILED or TIMED OUT (see $video_dir/eval.log)" | tee -a "$RESULTS_FILE"
        continue
    fi

    success_line="$(grep 'Total success rate' "$video_dir/eval.log" | tail -1)"
    echo "$run_name: $success_line" | tee -a "$RESULTS_FILE"
done

echo
echo "=== Summary (also written to $RESULTS_FILE) ==="
cat "$RESULTS_FILE"
