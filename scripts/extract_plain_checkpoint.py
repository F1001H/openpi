#!/usr/bin/env python3
"""Extracts servable, flat/device-count-agnostic checkpoints from a PLAIN
scripts/train.py run (no OpenPIWithJEPA wrapper) -- e.g. plain_control or
any of the data-fraction sweep runs.

WHY THIS EXISTS, DISTINCT FROM extract_base_model_checkpoint.py: that
script assumes every pi05_libero checkpoint went through train_end_to_end.py
(OpenPIWithJEPA-wrapped, top-level keys base_model/jepa_predictor/
target_norm/...) and unconditionally does
`train_state.params.to_pure_dict()["base_model"]` to strip the wrapper.
Every full-finetune run in this project WAS trained that way -- except
plain_control and the data-fraction sweep, which use plain scripts/train.py
directly and therefore have NO wrapper: the on-disk checkpoint's params ARE
the bare Pi0Config-shaped tree already. Pointing extract_base_model_
checkpoint.py at one of these fails immediately on restore: it builds its
abstract restore shape via train_end_to_end.py's init_train_state (which
constructs an OpenPIWithJEPA-wrapped model), and orbax's structure check
rejects it against the real (unwrapped, and simply SMALLER -- no
jepa_predictor/vision_proj/target_norm at all) on-disk tree before ever
reaching the ["base_model"] indexing that would also have failed.
Confirmed via a real crash extracting plain_control's checkpoint this way.

Same restore-flatten-resave purpose otherwise (see extract_base_model_
checkpoint.py's docstring for the underlying reason this step exists at
all): the raw checkpoint is N-way FSDP-sharded and can't be restored on
fewer real devices without this.

Usage:
    uv run scripts/extract_plain_checkpoint.py <config_name> \
        --exp-name=<exp_name> --out-dir=/path/to/servable_checkpoints \
        [--steps=5000,10000] [--checkpoint-base-dir=./checkpoints] \
        [--fsdp-devices=1]

Then serve/evaluate any one of them exactly like extract_base_model_
checkpoint.py's output:
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=<config_name> --policy.dir=/path/to/servable_checkpoints/<step>
"""

import argparse
import logging
import shutil
import sys

import etils.epath as epath
import jax
import orbax.checkpoint as ocp

import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding

from train import init_logging, init_train_state


def _extract_one(
    config: _config.TrainConfig,
    checkpoint_manager,
    train_state_shape,
    step: int,
    out_root: epath.Path,
) -> None:
    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, None, step=step)
    logging.info(f"Restored checkpoint step={int(train_state.step)} (directory {step}) from {config.checkpoint_dir}")

    # No wrapper to strip -- plain scripts/train.py's params ARE the bare
    # model tree already (contrast extract_base_model_checkpoint.py's
    # ["base_model"] indexing, which would KeyError here even if the
    # abstract-shape restore hadn't already failed first).
    model_params = train_state.params.to_pure_dict()

    out_path = out_root / str(step)
    out_path.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(str(out_path / "params"), {"params": model_params})

    src_assets = epath.Path(config.checkpoint_dir) / str(step) / "assets"
    if src_assets.exists():
        shutil.copytree(str(src_assets), str(out_path / "assets"), dirs_exist_ok=True)
    else:
        logging.warning(f"No assets/ found at {src_assets} -- norm_stats won't be bundled with {out_path}.")

    logging.info(f"Wrote servable checkpoint (step {step}) to {out_path}")


def main(config: _config.TrainConfig, steps: list[int] | None, out_dir: str, fsdp_devices: int) -> None:
    init_logging()
    mesh = sharding.make_mesh(fsdp_devices)
    rng = jax.random.key(config.seed)

    train_state_shape, _ = init_train_state(config, rng, mesh, resume=True)
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True,
    )
    if not resuming:
        raise RuntimeError(f"No checkpoint found at {config.checkpoint_dir}.")

    target_steps = steps if steps is not None else sorted(checkpoint_manager.all_steps())
    if not target_steps:
        raise RuntimeError(f"No checkpoint steps found at {config.checkpoint_dir}.")
    logging.info(f"Extracting {len(target_steps)} step(s): {target_steps}")

    out_root = epath.Path(out_dir)
    for step in target_steps:
        _extract_one(config, checkpoint_manager, train_state_shape, step, out_root)

    logging.info(
        f"Done. Serve any step with: uv run scripts/serve_policy.py policy:checkpoint "
        f"--policy.config={config.name} --policy.dir={out_dir}/<step>"
    )


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument(
        "--steps", type=str, default=None,
        help="Comma-separated checkpoint steps to extract (default: every step CheckpointManager still has).",
    )
    _parser.add_argument("--out-dir", type=str, required=True)
    _parser.add_argument(
        "--fsdp-devices", type=int, default=1,
        help="Must match the --fsdp-devices the checkpoint was trained with.",
    )
    _args, _remaining = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    _steps = [int(s) for s in _args.steps.split(",")] if _args.steps else None
    main(_config.cli(), _steps, _args.out_dir, _args.fsdp_devices)
