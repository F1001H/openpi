#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""
Consolidated entrypoint: init_logging / init_wandb are unchanged from the
very first version of this script. What's new here is main()'s data loading,
wired to JepaTransitionDataLoader instead of _data_loader.create_data_loader,
and batch handling updated for the (obs_t, action_chunk, obs_t1) triple that
train_step_transitions.train_step expects.

CORRECTED (v2): earlier versions of this file invented top-level TrainConfig
fields (lerobot_root, lerobot_repo_id, action_horizon, data_num_workers) that
don't actually exist on your real TrainConfig. Now that I have config.py:
    - dataset root/repo_id come from config.data (a DataConfigFactory, e.g.
      KoboDataConfig) via config.data.create(...) -- set with the *real*
      --data.root=... CLI override, not a custom flag.
    - action_horizon is config.model.action_horizon (already exists on
      BaseModelConfig/Pi0Config).
    - num_workers is config.num_workers (already exists on TrainConfig).
    - jepa_predictor_checkpoint is NOT threaded through TrainConfig at all
      (avoids touching your real config.py) -- it's now a plain keyword
      argument to main() / init_train_state().

HEADS UP -- bug in your config.py, independent of anything here:
DataConfigFactory.create_base_config() always sets `root=self.root`,
unconditionally overwriting whatever root was set on `self.base_config`. Your
pi0_kobo_cube config sets root via `base_config=DataConfig(root=...)`, which
this silently discards -- the only way to actually get a local root through
right now is `--data.root=/path` on the CLI. Worth fixing independent of this
JEPA work since it likely affects your normal BC-only runs too.

STILL OPEN / NOT ADDRESSED HERE (see prior discussion):
    - data-loader resume/checkpoint state: `_checkpoints.restore_state` is
      still called with `jepa_loader` in the same slot the old `data_loader`
      occupied, but JepaTransitionDataLoader has no serializable position --
      resuming restarts the episode-shuffle stream from scratch rather than
      picking up where it left off.
    - raw_batch_to_transition's prompt_tokens=None placeholder in
      lerobot_v3_transition_loader.py -- still needs your real tokenizer
      wired in before this runs for real.
"""

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

from jepa.train_step_transitions import OpenPIWithJEPA, train_step
from utils.data_loader import JepaTransitionDataLoader


# =========================================================================== #
# Logging / wandb / checkpoint init -- unchanged from the original script
# =========================================================================== #

def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)
    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """params_shape is OpenPIWithJEPA's FULL params shape (top-level keys:
    base_model, jepa_predictor, target_norm). Pretrained checkpoints (e.g.
    pi05_base) were saved from a plain, unwrapped Pi0Config model -- their
    stored pytree has no "base_model" prefix at all. Handing the loader the
    full wrapped shape made it try to restore into a structure the checkpoint
    was never saved with, producing a 0-child result and a pytree-structure
    mismatch at check_pytree_equality.

    Fix: restore into the base_model sub-shape only, then re-nest the
    (possibly partial/sparse -- see the ShapeDtypeStruct filtering below,
    which already produces incomplete dicts for LoRA-style partial
    checkpoints) result under "base_model". state.replace_by_pure_dict
    merges this in, leaving jepa_predictor/target_norm at their freshly
    initialized values since they're simply absent from what's merged.
    """
    base_model_shape = params_shape["base_model"]
    loaded_params = loader.load(base_model_shape)
    at.check_pytree_equality(expected=base_model_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    cleaned = traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )
    return {"base_model": cleaned}


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool,
    jepa_predictor_checkpoint: str | None = None,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        base_model = config.model.create(model_rng)

        rng, jepa_rng = jax.random.split(rng)
        model = OpenPIWithJEPA(base_model, config, nnx.Rngs(jepa_rng))

        if partial_params is not None:
            # NOTE (was: state.replace_by_pure_dict(partial_params)): that
            # call, given a SPARSE dict (only "base_model" present, since
            # _load_weights_and_validate deliberately omits jepa_predictor/
            # target_norm entirely), appears to rebuild the state's
            # structure to match only what's provided rather than merging
            # values into the existing structure -- observed as a genuine
            # nnx.graphdef(model) mismatch between this branch (only taken
            # in the real jax.jit(init,...) call, since jax.eval_shape calls
            # init with partial_params=None) and the eval_shape pass, which
            # never takes this branch. Root cause not fully confirmed against
            # NNX's replace_by_pure_dict source, but the fix below sidesteps
            # the question entirely: per-leaf `.value = ...` mutation (the
            # same pattern load_and_merge_predictor_state already uses) can
            # only change array VALUES at paths that already exist in the
            # graph -- it cannot add or remove graph nodes, so it cannot
            # reproduce this failure mode regardless of the exact cause.
            graphdef, state = nnx.split(model)
            pure = state.to_pure_dict()
            flat_partial = traverse_util.flatten_dict(partial_params, sep=".")
            missing = []
            for flat_key, value in flat_partial.items():
                parts = flat_key.split(".")
                d = pure
                try:
                    for p in parts[:-1]:
                        d = d[p] if p in d else d[int(p)]
                    leaf_key = parts[-1]
                    target = d[leaf_key] if leaf_key in d else d[int(leaf_key)]
                    target.value = jnp.asarray(value)
                except (KeyError, IndexError):
                    missing.append(flat_key)
            if missing:
                raise RuntimeError(
                    f"{len(missing)} keys from the pretrained weight loader did not map onto the model "
                    f"(e.g. {missing[:5]}). This usually means config.model doesn't match the checkpoint's "
                    f"architecture, or the base_model re-nesting in _load_weights_and_validate needs adjusting."
                )
            model = nnx.merge(graphdef, pure)

        if jepa_predictor_checkpoint is not None:
            # Merge pretrained V-JEPA2-AC predictor weights converted by
            # convert_checkpoint.py. Must happen AFTER partial_params merge
            # above (so it isn't overwritten) and BEFORE nnx.state(model)
            # below (so the loaded values are what gets frozen/bf16-cast and
            # captured as the initial params).
            # NOTE: unlike partial_params (base pi0.5 weights, gated so they
            # only load during the real jax.jit(init,...) call, not during
            # the jax.eval_shape(init, init_rng) shape-inference pass below),
            # this branch isn't gated the same way, so the (much smaller)
            # predictor .npz gets read twice -- once for shape inference,
            # once for real. Harmless for a modest predictor checkpoint;
            # worth gating identically if this npz ever gets large.
            from jepa.convert_checkpoint import load_and_merge_predictor_state
            model = load_and_merge_predictor_state(model, jepa_predictor_checkpoint)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# =========================================================================== #
# main() -- updated for JepaTransitionDataLoader / 3-tuple batch
# =========================================================================== #

def main(config: _config.TrainConfig, jepa_predictor_checkpoint: str | None = None):
    init_logging()
    logging.info(f"Running on computational cluster node: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # --- Data loading: JepaTransitionDataLoader instead of _data_loader.create_data_loader --- #
    # Resolve root/repo_id the same way the rest of openpi does: through
    # config.data (a DataConfigFactory), not invented top-level TrainConfig
    # fields. NOTE the create_base_config bug described in the module
    # docstring -- root only comes through if you passed --data.root=... on
    # the CLI, regardless of what's baked into the config registry entry.
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.root is not None:
        repo_id_or_root = str(data_config.root)
        is_local_root = True
    elif data_config.repo_id is not None:
        if data_config.repo_id.startswith("local/"):
            # KoboDataConfig-style configs set repo_id to a local-only
            # placeholder string (e.g. "local/bimanual_cube") that isn't a
            # real Hub id. Treating it as one would 404 against the Hub with
            # a confusing error instead of telling you what's actually wrong.
            raise ValueError(
                f"config.data.repo_id ('{data_config.repo_id}') looks like a local-only placeholder, "
                f"not a real Hub repo id, and config.data.root is None. Pass --data.root=/path/to/dataset "
                f"on the CLI (see the create_base_config bug noted in this file's module docstring)."
            )
        repo_id_or_root = data_config.repo_id
        is_local_root = False
    else:
        raise ValueError("config.data resolved to neither a root path nor a repo_id.")

    action_horizon = config.model.action_horizon
    num_workers = config.num_workers

    jepa_loader = JepaTransitionDataLoader(
        config=config,
        repo_id_or_root=repo_id_or_root,
        action_horizon=action_horizon,
        data_sharding=data_sharding,
        batch_size=config.batch_size,
        num_workers=num_workers,
        is_local_root=is_local_root,
        seed=config.seed,
    )
    data_iter = iter(jepa_loader)
    batch = next(data_iter)  # (obs_t, action_chunk, obs_t1), already device_put onto data_sharding
    obs_t, action_chunk, obs_t1 = batch
    logging.info(f"Initialized data loader structure:\n{training_utils.array_tree_to_info(batch)}")

    # Log initial camera images from obs_t (unchanged: obs_t.images is still a dict of camera views).
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in obs_t.images.values()], axis=1))
        for i in range(min(5, len(next(iter(obs_t.images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming, jepa_predictor_checkpoint=jepa_predictor_checkpoint
    )
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state parameters footprint:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        # NOTE: still-open item -- JepaTransitionDataLoader has no
        # serializable iteration position, so this restores model/optimizer
        # state correctly but the data stream itself restarts from the
        # beginning of its (seeded) episode shuffle, not from wherever
        # training had gotten to. Revisit if restore_state's actual contract
        # requires more than that from `data_loader`.
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, jepa_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    try:
        for step in pbar:
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
            infos.append(info)

            if step % config.log_interval == 0:
                stacked_infos = common_utils.stack_forest(infos)
                reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(reduced_info, step=step)
                infos = []
            batch = next(data_iter)

            if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
                _checkpoints.save_state(checkpoint_manager, train_state, jepa_loader, step)
    finally:
        jepa_loader.close()

    logging.info("Terminating process iterations. Flushing checkpoint queues...")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    import argparse
    import sys

    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--jepa-predictor-checkpoint", type=str, default=None)
    _args, _remaining = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    main(_config.cli(), jepa_predictor_checkpoint=_args.jepa_predictor_checkpoint)