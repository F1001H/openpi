#!/usr/bin/env python3
"""Extracts servable, plain Pi0Config-shaped checkpoints from a JEPA
co-training checkpoint (scripts/train_end_to_end.py's OpenPIWithJEPA wrapper)
-- one per surviving checkpoint step, not just the latest.

WHY THIS EXISTS: scripts/serve_policy.py's checkpoint loader
(policy_config.create_trained_policy -> BaseModel.load ->
ocp.transform_utils.intersect_trees) expects a plain, unwrapped
train_config.model (e.g. Pi0Config)-shaped params pytree -- no "base_model"
prefix. Checkpoints saved by train_end_to_end.py are OpenPIWithJEPA-shaped
instead (top-level keys base_model/jepa_predictor/target_norm, see
train_end_to_end.py's _load_weights_and_validate docstring for the same
issue in the OPPOSITE direction). Pointing --policy.dir straight at a
JEPA-cotraining checkpoint dir fails: intersect_trees finds no overlapping
paths between "base_model.PaliGemma...." and the plain model's
"PaliGemma...." (no prefix), so the params it hands to BaseModel.load are
empty/wrong and the subsequent equality check raises.

Restores the FULL wrapped train_state the same way qc_label_rewards.py does
(train_end_to_end.init_train_state(..., resume=True) + restore_state), pulls
out just params["base_model"] (already the complete, EMA-blended-with-frozen
pytree -- see checkpoints.py's _split_params, which reconstructs the full
tree at SAVE time, so nothing extra needs merging here), and writes it out
in the flat {params/, assets/} layout create_trained_policy/--policy.dir
expects (matching how released checkpoints like
gs://openpi-assets/checkpoints/pi05_libero/ are laid out -- NOT the
per-training-step-nested layout your own checkpoint_base_dir uses).

By default extracts EVERY step CheckpointManager still has on disk (see
checkpoints.py's initialize_checkpoint_dir: max_to_keep=1 + keep_period
means a periodic trail of steps survives, e.g. {5000,10000,...,25000,29999}
for a 30k-step run with the default keep_period=5000 -- NOT just the
latest), one subdirectory per step under --out-dir (--out-dir/<step>/
{params,assets}). Pass --steps to restrict to a subset instead.

Usage:
    uv run scripts/extract_base_model_checkpoint.py <config_name> \
        --exp-name=<exp_name> --out-dir=/path/to/servable_checkpoints \
        [--steps=5000,10000] [--checkpoint-base-dir=./checkpoints]

Then serve/evaluate any one of them with:
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

    base_model_params = train_state.params.to_pure_dict()["base_model"]

    out_path = out_root / str(step)
    out_path.mkdir(parents=True, exist_ok=True)
    # Matches restore_params' expected on-disk shape (model.py:289) -- a
    # PyTreeCheckpointer save of {"params": <pytree>}, read back at
    # "<checkpoint_dir>/params".
    ocp.PyTreeCheckpointer().save(str(out_path / "params"), {"params": base_model_params})

    # Copy the SAME assets/ (norm_stats) this step's training checkpoint
    # already wrote -- unaffected by the base_model extraction, so just
    # reused as-is rather than recomputed.
    src_assets = epath.Path(config.checkpoint_dir) / str(step) / "assets"
    if src_assets.exists():
        shutil.copytree(str(src_assets), str(out_path / "assets"), dirs_exist_ok=True)
    else:
        logging.warning(f"No assets/ found at {src_assets} -- norm_stats won't be bundled with {out_path}.")

    logging.info(f"Wrote servable checkpoint (step {step}) to {out_path}")


def main(config: _config.TrainConfig, steps: list[int] | None, out_dir: str) -> None:
    init_logging()
    mesh = sharding.make_mesh(1)
    rng = jax.random.key(config.seed)

    # Shape inference is cheap (jax.eval_shape, no real I/O) -- build once,
    # reuse across every step's restore below.
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
    _args, _remaining = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    _steps = [int(s) for s in _args.steps.split(",")] if _args.steps else None
    main(_config.cli(), _steps, _args.out_dir)
