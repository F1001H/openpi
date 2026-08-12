#!/bin/bash
# Runs the LIBERO sim eval (examples/libero/main.py) against extracted sweep
# checkpoints, plain-BC (scripts/serve_policy.py, NOT the QC critic-augmented
# path -- this compares the alpha_bc/beta_jepa sweep points themselves;
# QC/critic eval is a separate follow-up on whichever checkpoint wins here).
# One checkpoint at a time: starts the server, waits for it to come up, runs
# the sim rollouts, records the success rate, kills the server, moves on.
#
# Prerequisites (see this session's earlier setup):
#   - Each checkpoint already extracted via scripts/extract_base_model_checkpoint.py
#     and rsynced locally into CHECKPOINTS_ROOT, one subdir per sweep point.
#   - examples/libero/.venv set up (git submodule + the non-Docker README steps).
#
# Usage:
#   ./scripts/eval_libero_sweep_local.sh <checkpoints_root_dir> [task_suite_name] [num_trials_per_task]
#   ALL_STEPS=true ./scripts/eval_libero_sweep_local.sh <checkpoints_root_dir> [task_suite_name] [num_trials_per_task]
#
# Example:
#   ./scripts/eval_libero_sweep_local.sh ~/openpi_eval_checkpoints libero_10 20
#   ALL_STEPS=true ./scripts/eval_libero_sweep_local.sh ~/openpi_eval_checkpoints libero_10 20
#
# checkpoints_root_dir must contain one subdir per sweep point, each in turn
# containing one subdir per extracted step (scripts/extract_base_model_checkpoint.py
# now extracts EVERY surviving checkpoint step by default, not just the
# latest), e.g.:
#   ~/openpi_eval_checkpoints/beta0.0/5000/{params,assets}
#   ~/openpi_eval_checkpoints/beta0.0/10000/{params,assets}
#   ...
#   ~/openpi_eval_checkpoints/beta0.0/29999/{params,assets}
#   ~/openpi_eval_checkpoints/beta0.1/5000/{params,assets}
#   ...
#
# Default (ALL_STEPS unset/false) evaluates only each sweep point's LATEST
# (highest-step) checkpoint -- the fully-trained one, which is what picking
# a sweep winner needs. ALL_STEPS=true evaluates every numbered step subdir
# under every sweep point instead (e.g. for a training-curve plot) -- this
# multiplies total run time by however many steps survived per run (~6 for
# a 30k-step run with the default keep_period=5000), so budget accordingly.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHECKPOINTS_ROOT="${1:?Usage: $0 <checkpoints_root_dir> [task_suite_name] [num_trials_per_task]}"
TASK_SUITE="${2:-libero_10}"
NUM_TRIALS="${3:-20}"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_low_mem}"
PORT="${PORT:-8000}"
ALL_STEPS="${ALL_STEPS:-false}"

LIBERO_VENV="examples/libero/.venv"
export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"
export LIBERO_CONFIG_PATH="$PWD/third_party/libero/.libero_config"

RESULTS_FILE="${RESULTS_FILE:-libero_sweep_eval_results.txt}"
: > "$RESULTS_FILE"

eval_one_checkpoint() {
    local name="$1"
    local ckpt_dir="$2"

    echo "=== Evaluating $name ($ckpt_dir) ==="
    local video_dir="libero_eval_videos/$name"
    mkdir -p "$video_dir"

    # --port must come BEFORE the policy:checkpoint subcommand token --
    # tyro's subcommand parsing doesn't accept top-level Args fields (like
    # --port) after it (confirmed: passing it after silently fails with
    # "Unrecognized options", which left the eval client waiting forever
    # against a server that never started).
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py --port="$PORT" policy:checkpoint \
        --policy.config="$CONFIG_NAME" --policy.dir="$ckpt_dir" \
        > "$video_dir/server.log" 2>&1 &
    local server_pid=$!

    # Wait for the server to actually start listening, rather than a fixed
    # sleep -- checkpoint restore time varies. Also checks server_pid is
    # still alive: if the server crashed/exited immediately, don't wait the
    # full timeout -- fail fast instead of letting the eval client hang
    # indefinitely against a server that will never come up.
    local server_up=false
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
        echo "$name: SERVER FAILED TO START (see $video_dir/server.log)" | tee -a "$RESULTS_FILE"
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        return
    fi

    set +e
    (
        source "$LIBERO_VENV/bin/activate"
        MUJOCO_GL=egl timeout 3600 python examples/libero/main.py \
            --args.port "$PORT" \
            --args.task-suite-name "$TASK_SUITE" --args.num-trials-per-task "$NUM_TRIALS" \
            --args.video-out-path "$video_dir"
    ) > "$video_dir/eval.log" 2>&1
    local eval_status=$?
    set -e

    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true

    if [ $eval_status -ne 0 ]; then
        echo "$name: EVAL FAILED or TIMED OUT (see $video_dir/eval.log)" | tee -a "$RESULTS_FILE"
        return
    fi

    local success_line
    success_line="$(grep 'Total success rate' "$video_dir/eval.log" | tail -1)"
    echo "$name: $success_line" | tee -a "$RESULTS_FILE"
}

for sweep_point_dir in "$CHECKPOINTS_ROOT"/*/; do
    sweep_point_name="$(basename "$sweep_point_dir")"

    # Numeric sort, not lexicographic (lexicographic would put "5000" after
    # "29999").
    step_names=()
    for step_dir in "$sweep_point_dir"*/; do
        step_name="$(basename "$step_dir")"
        case "$step_name" in
            ''|*[!0-9]*) continue ;;  # not a plain integer step dir (e.g. leftover params/assets)
        esac
        [ -d "${step_dir}params" ] || continue
        step_names+=("$step_name")
    done
    if [ ${#step_names[@]} -eq 0 ]; then
        echo "Skipping $sweep_point_name (no numbered step subdir with a params/ dir found under $sweep_point_dir)"
        continue
    fi
    IFS=$'\n' sorted_steps=($(sort -n <<<"${step_names[*]}")); unset IFS

    if [ "$ALL_STEPS" = true ]; then
        target_steps=("${sorted_steps[@]}")
    else
        target_steps=("${sorted_steps[-1]}")  # latest only
    fi

    for step in "${target_steps[@]}"; do
        eval_one_checkpoint "${sweep_point_name}_step${step}" "$sweep_point_dir$step/"
    done
done

echo
echo "=== Summary (also written to $RESULTS_FILE) ==="
cat "$RESULTS_FILE"
