"""Local end-to-end smoke test: runs the REAL main() from train_main_jepa.py,
against your REAL dataset, but with a drastically shrunk config so it
completes in minutes on a laptop/CPU instead of needing the cluster.

WHY REUSE main() RATHER THAN WRITE A SEPARATE MINI-HARNESS: the whole value
of a smoke test is catching integration bugs -- shape mismatches between the
predictor and your actual encoder, the raw_batch_to_transition ADAPT stub
blowing up on your actual Observation dataclass, nnx.gelu/silu not existing,
jit compilation failing, checkpoint dry-run mismatches, etc. A hand-rolled
mini version of the pipeline would risk diverging from the real path and
passing while the real thing still breaks. This runs the identical code path
at tiny scale instead.

CORRECTED (v2): earlier versions of this script invented --lerobot-root /
--action-horizon flags backed by nonexistent TrainConfig fields. Root/repo_id
and action_horizon are real, existing parts of your config -- root via the
DataConfigFactory's native --data.root CLI override (see the bug noted in
train_main_jepa.py's docstring: this is the ONLY way root currently gets
through, since create_base_config discards whatever's baked into the config
registry entry), and action_horizon via config.model.action_horizon. Nothing
extra needed from this script for either -- just pass --data.root on the CLI
like you would for a normal training run.

WHAT THIS DELIBERATELY DOES NOT TEST:
  - Actual FSDP/multi-device sharding behavior (fsdp_devices forced to 1).
  - Real training dynamics / whether the loss actually goes down over a
    meaningful number of steps (num_train_steps is tiny).
  - Multi-worker data loading edge cases (num_workers forced to 0 -- so the
    multi-worker episode-sharding path in the data loader is NOT exercised
    here; run test_transition_loader.py --num-workers>1 separately for that).
  - wandb logging (disabled) and real checkpoint resume (skipped).

USAGE:
    JAX_PLATFORMS=cpu uv run scripts/test_train.py pi0_kobo_cube \\
        --exp-name smoke_test_1 \\
        --data.root /home/fabian/kobo_cube

If you have a converted pretrained JEPA predictor checkpoint (from
convert_checkpoint.py --save-path=...), pass it too:
    --jepa-predictor-checkpoint /path/to/converted_predictor.npz

Without JAX_PLATFORMS set, this uses whatever device JAX finds first (a
local GPU if you have one) -- CPU is just the "definitely available
everywhere, definitely not touching the cluster" recommendation, not a hard
requirement.
"""

import argparse
import dataclasses
import os
import sys

# Must happen before `import jax` anywhere (including transitively via the
# imports below) for JAX_PLATFORMS to take effect. If you already export
# JAX_PLATFORMS=cpu in your shell, this is a no-op.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import openpi.training.config as _config
from train_end_to_end import main


SMOKE_SCALE_OVERRIDES = dict(
    log_interval=1,
    save_interval=10**9,  # effectively never -- don't exercise checkpoint-saving in a smoke test
    wandb_enabled=False,
    fsdp_devices=1,
    num_workers=0,         # single-process data loading -- easier to get a real traceback from
    overwrite=True,        # smoke-test checkpoint dir gets clobbered every run, that's fine
)


def build_smoke_config(base_config, num_steps: int, batch_size: int) -> _config.TrainConfig:
    """All of these are real, existing TrainConfig fields (see config.py),
    so unlike the earlier version of this function, nothing here should ever
    need to be silently skipped."""
    overrides = dict(SMOKE_SCALE_OVERRIDES, num_train_steps=num_steps, batch_size=batch_size)
    return dataclasses.replace(base_config, **overrides)


def main_cli():
    parser = argparse.ArgumentParser(add_help=False)  # let _config.cli() own --help
    parser.add_argument("--num-steps", type=int, default=3,
                         help="How many train_step calls to run -- just enough to confirm it compiles and runs, not to train anything")
    parser.add_argument("--batch-size", type=int, default=2,
                         help="Keep this small -- CPU forward+backward through a full ViT-scale predictor "
                              "at a large batch size will be slow. This is a compile/shape/NaN check, not a "
                              "throughput benchmark.")
    parser.add_argument("--jepa-predictor-checkpoint", type=str, default=None,
                         help="Path to a converted .npz from convert_checkpoint.py --save-path=..., "
                              "if you want the smoke test to exercise pretrained-weight loading too. "
                              "Omitting this trains the predictor from random init instead -- still a "
                              "valid structural smoke test, just doesn't check the weight conversion path.")
    args, remaining_argv = parser.parse_known_args()

    # Reuse the real --config-name / --data.root / --exp-name / etc. CLI
    # mechanism your real entrypoint uses, rather than guessing at any of it
    # here. `remaining_argv` is whatever's left after stripping the
    # smoke-test-specific flags above (which don't exist on TrainConfig).
    sys.argv = [sys.argv[0]] + remaining_argv
    base_config = _config.cli()

    config = build_smoke_config(base_config, args.num_steps, args.batch_size)

    data_config = config.data.create(config.assets_dirs, config.model)
    print("=" * 70)
    print("SMOKE TEST -- reduced config:")
    print(f"  config.name={config.name}  exp_name={config.exp_name}")
    print(f"  num_train_steps={config.num_train_steps}  batch_size={config.batch_size}  "
          f"fsdp_devices={config.fsdp_devices}  num_workers={config.num_workers}")
    print(f"  data.root={data_config.root}  data.repo_id={data_config.repo_id}")
    print(f"  model.action_horizon={config.model.action_horizon}")
    print(f"  jepa_predictor_checkpoint={args.jepa_predictor_checkpoint}")
    if data_config.root is None:
        print("  WARNING: data.root is None -- did you forget --data.root=/path/to/dataset ? "
              "(see the create_base_config bug noted in train_main_jepa.py's docstring)")
    print("=" * 70)

    main(config, jepa_predictor_checkpoint=args.jepa_predictor_checkpoint)

    print("\nSmoke test completed without raising -- this does NOT mean training is correct, "
          "only that the pipeline runs end-to-end: data loads, shapes line up, jit compiles, "
          "gradients flow, and no exception was silently swallowed. Check the loss values printed "
          "above for obviously wrong signs of life (e.g. NaN, or loss_jepa suspiciously at exactly "
          "0 -- the latter would suggest the same-observation-target bug is back).")


if __name__ == "__main__":
    main_cli()