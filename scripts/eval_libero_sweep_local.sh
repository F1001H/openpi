#!/bin/bash
# Runs the LIBERO sim eval (examples/libero/main.py) against each extracted
# sweep checkpoint in turn, plain-BC (scripts/serve_policy.py, NOT the QC
# critic-augmented path -- this compares the alpha_bc/beta_jepa sweep
# points themselves; QC/critic eval is a separate follow-up on whichever
# checkpoint wins here). One checkpoint at a time: starts the server,
# waits for it to come up, runs the sim rollouts, records the success rate,
# kills the server, moves to the next checkpoint dir.
#
# Prerequisites (see this session's earlier setup):
#   - Each checkpoint already extracted via scripts/extract_base_model_checkpoint.py
#     and rsynced locally into CHECKPOINTS_ROOT, one subdir per sweep point.
#   - examples/libero/.venv set up (git submodule + the non-Docker README steps).
#
# Usage:
#   ./scripts/eval_libero_sweep_local.sh <checkpoints_root_dir> [task_suite_name] [num_trials_per_task]
#
# Example:
#   ./scripts/eval_libero_sweep_local.sh ~/openpi_eval_checkpoints libero_10 20
#
# checkpoints_root_dir must contain one subdir per sweep point, e.g.:
#   ~/openpi_eval_checkpoints/beta0.0/{params,assets}
#   ~/openpi_eval_checkpoints/beta0.1/{params,assets}
#   ...

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHECKPOINTS_ROOT="${1:?Usage: $0 <checkpoints_root_dir> [task_suite_name] [num_trials_per_task]}"
TASK_SUITE="${2:-libero_10}"
NUM_TRIALS="${3:-20}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_low_mem}"
PORT="${PORT:-8000}"

LIBERO_VENV="examples/libero/.venv"
export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"
export LIBERO_CONFIG_PATH="$PWD/third_party/libero/.libero_config"

RESULTS_FILE="${RESULTS_FILE:-libero_sweep_eval_results.txt}"
: > "$RESULTS_FILE"

for ckpt_dir in "$CHECKPOINTS_ROOT"/*/; do
    name="$(basename "$ckpt_dir")"
    if [ ! -d "$ckpt_dir/params" ]; then
        echo "Skipping $name (no params/ subdir -- not an extracted checkpoint)"
        continue
    fi

    echo "=== Evaluating $name ($ckpt_dir) ==="
    video_dir="libero_eval_videos/$name"
    mkdir -p "$video_dir"

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config="$CONFIG_NAME" --policy.dir="$ckpt_dir" --port="$PORT" \
        > "libero_eval_videos/$name/server.log" 2>&1 &
    server_pid=$!

    # Wait for the server to actually start listening, rather than a fixed
    # sleep -- checkpoint restore time varies (first run pays JAX compilation
    # cache cost too).
    for _ in $(seq 1 120); do
        if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
            break
        fi
        sleep 2
    done

    set +e
    (
        source "$LIBERO_VENV/bin/activate"
        MUJOCO_GL=egl python examples/libero/main.py \
            --args.port "$PORT" \
            --args.task-suite-name "$TASK_SUITE" --args.num-trials-per-task "$NUM_TRIALS" \
            --args.video-out-path "$video_dir"
    ) > "libero_eval_videos/$name/eval.log" 2>&1
    eval_status=$?
    set -e

    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true

    if [ $eval_status -ne 0 ]; then
        echo "$name: EVAL FAILED (see libero_eval_videos/$name/eval.log)" | tee -a "$RESULTS_FILE"
        continue
    fi

    success_line="$(grep 'Total success rate' "libero_eval_videos/$name/eval.log" | tail -1)"
    echo "$name: $success_line" | tee -a "$RESULTS_FILE"
done

echo
echo "=== Summary (also written to $RESULTS_FILE) ==="
cat "$RESULTS_FILE"
