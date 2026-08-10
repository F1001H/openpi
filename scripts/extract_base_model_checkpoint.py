#!/usr/bin/env python3
"""Extracts a servable, plain Pi0Config-shaped checkpoint from a JEPA
co-training checkpoint (scripts/train_end_to_end.py's OpenPIWithJEPA wrapper).

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

Usage:
    uv run scripts/extract_base_model_checkpoint.py <config_name> \
        --exp-name=<exp_name> --out-dir=/path/to/servable_checkpoint \
        [--step=N] [--checkpoint-base-dir=./checkpoints]

Then serve/evaluate with:
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=<config_name> --policy.dir=/path/to/servable_checkpoint
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


def main(config: _config.TrainConfig, step: int | None, out_dir: str) -> None:
    init_logging()
    mesh = sharding.make_mesh(1)
    rng = jax.random.key(config.seed)

    train_state_shape, _ = init_train_state(config, rng, mesh, resume=True, jepa_predictor_checkpoint=None)
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True,
    )
    if not resuming:
        raise RuntimeError(f"No checkpoint found at {config.checkpoint_dir}.")
    # The on-disk checkpoint directory is named after the step CheckpointManager
    # was given at save time (train_end_to_end.py's 0-indexed loop counter),
    # NOT train_state.step after restore -- that field is the cumulative
    # completed-update count (e.g. 3 after 3 updates at loop steps 0,1,2),
    # off by one from the directory name ("2"). Use the manager's own
    # bookkeeping for the directory path, not the restored train_state.
    disk_step = step if step is not None else checkpoint_manager.latest_step()
    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, None, step=step)
    logging.info(f"Restored checkpoint step={int(train_state.step)} (directory {disk_step}) from {config.checkpoint_dir}")

    base_model_params = train_state.params.to_pure_dict()["base_model"]

    out_path = epath.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # Matches restore_params' expected on-disk shape (model.py:289) -- a
    # PyTreeCheckpointer save of {"params": <pytree>}, read back at
    # "<checkpoint_dir>/params".
    ocp.PyTreeCheckpointer().save(str(out_path / "params"), {"params": base_model_params})

    # Copy the SAME assets/ (norm_stats) this step's training checkpoint
    # already wrote -- unaffected by the base_model extraction, so just
    # reused as-is rather than recomputed.
    src_assets = epath.Path(config.checkpoint_dir) / str(disk_step) / "assets"
    if src_assets.exists():
        shutil.copytree(str(src_assets), str(out_path / "assets"), dirs_exist_ok=True)
    else:
        logging.warning(f"No assets/ found at {src_assets} -- norm_stats won't be bundled with {out_dir}.")

    logging.info(f"Wrote servable checkpoint (step {disk_step}) to {out_dir}")
    logging.info(
        f"Serve with: uv run scripts/serve_policy.py policy:checkpoint "
        f"--policy.config={config.name} --policy.dir={out_dir}"
    )


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--step", type=int, default=None, help="Checkpoint step to restore (default: latest).")
    _parser.add_argument("--out-dir", type=str, required=True)
    _args, _remaining = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    main(_config.cli(), _args.step, _args.out_dir)
