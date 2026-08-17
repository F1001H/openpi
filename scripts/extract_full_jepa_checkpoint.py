#!/usr/bin/env python3
"""Extracts a servable, device-count-agnostic checkpoint of the FULL
OpenPIWithJEPA-wrapped model (base_model + jepa_predictor + vision_proj +
action_proj + state_proj + target_norm) from a JEPA co-training checkpoint --
one per surviving checkpoint step, not just the latest.

WHY THIS EXISTS: extract_base_model_checkpoint.py strips a training
checkpoint down to just `base_model`, which is all scripts/serve_policy.py's
plain-BC serving path needs -- but scripts/serve_qc_policy.py's critic-scored
best-of-N eval needs the WHOLE wrapped model (jepa_predictor/vision_proj
specifically: best_of_n_action_batch scores candidate actions using JEPA
embeddings, via OpenPIWithJEPA.extract_vision_latents). Pointing
serve_qc_policy.py straight at a raw multi-GPU-FSDP training checkpoint dir
means every local eval run needs that many real GPUs just to restore it
(sharding.make_mesh(fsdp_devices) requires that many devices actually
present) -- fine for the LoRA sweep (trained on 1 GPU), a hard blocker for
the full-finetune runs (trained on 4-way FSDP).

This script does the SAME restore-under-N-devices-then-flat-resave trick
extract_base_model_checkpoint.py already uses for base_model (proven to
work: those extracted checkpoints ARE restorable on 1 GPU via
serve_policy.py), just keeping the full params tree instead of slicing out
one subtree. The N-GPU restore step is unavoidable and must run wherever
--fsdp-devices real devices are available (i.e. on the cluster, matching
the training job's own --fsdp-devices) -- but it only has to happen ONCE per
step; the output is then rsync-able and servable anywhere with 1 GPU via
serve_qc_policy.py's --checkpoint-dir flag.

Usage (on the cluster, matching the training run's real GPU count):
    uv run scripts/extract_full_jepa_checkpoint.py <config_name> \
        --exp-name=<exp_name> --out-dir=/path/to/servable_jepa_checkpoints \
        [--steps=5000,10000] [--checkpoint-base-dir=./checkpoints] \
        [--fsdp-devices=4]

Then serve/evaluate any one of them LOCALLY (1 GPU) with:
    uv run scripts/serve_qc_policy.py <config_name> \
        --checkpoint-dir=/path/to/servable_jepa_checkpoints/<step> \
        --critic-checkpoint-path=/path/to/critic/final
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

from train_end_to_end import init_logging, init_train_state


def _extract_one(
    config: _config.TrainConfig,
    checkpoint_manager,
    train_state_shape,
    step: int,
    out_root: epath.Path,
) -> None:
    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, None, step=step)
    logging.info(f"Restored checkpoint step={int(train_state.step)} (directory {step}) from {config.checkpoint_dir}")

    # Unlike extract_base_model_checkpoint.py's ["base_model"] slice, keep
    # the WHOLE pure dict -- base_model/jepa_predictor/vision_proj/
    # action_proj/state_proj/target_norm all present, exactly OpenPIWithJEPA's
    # own top-level attribute structure.
    full_params = train_state.params.to_pure_dict()

    out_path = out_root / str(step)
    out_path.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(str(out_path / "params"), {"params": full_params})

    src_assets = epath.Path(config.checkpoint_dir) / str(step) / "assets"
    if src_assets.exists():
        shutil.copytree(str(src_assets), str(out_path / "assets"), dirs_exist_ok=True)
    else:
        logging.warning(f"No assets/ found at {src_assets} -- norm_stats won't be bundled with {out_path}.")

    logging.info(f"Wrote servable full-JEPA checkpoint (step {step}) to {out_path}")


def main(config: _config.TrainConfig, steps: list[int] | None, out_dir: str, fsdp_devices: int) -> None:
    init_logging()
    mesh = sharding.make_mesh(fsdp_devices)
    rng = jax.random.key(config.seed)

    train_state_shape, _ = init_train_state(config, rng, mesh, resume=True, jepa_predictor_checkpoint=None)
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
        f"Done. Serve any step locally (1 GPU) with: uv run scripts/serve_qc_policy.py {config.name} "
        f"--checkpoint-dir={out_dir}/<step> --critic-checkpoint-path=<critic/final>"
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
        help="Must match the --fsdp-devices the checkpoint was trained with (see module docstring).",
    )
    _args, _remaining = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    _steps = [int(s) for s in _args.steps.split(",")] if _args.steps else None
    main(_config.cli(), _steps, _args.out_dir, _args.fsdp_devices)
