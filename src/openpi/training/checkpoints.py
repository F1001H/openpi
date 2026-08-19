from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
from flax import nnx
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def _prune_to_structure(actual: nnx.State, expected: nnx.State) -> nnx.State:
    """Returns a copy of `actual` restricted to exactly the leaf paths
    present in `expected` -- for reconciling a restored State that may carry
    EXTRA leaves (e.g. a checkpoint saved before the _split_params fix
    above, which has non-Param buffers baked into its saved ema_params) back
    down to what the current abstract shape expects. Operates on
    flat_state()'s own key representation directly, never round-tripping
    through to_pure_dict()/replace_by_pure_dict -- that path's
    try_convert_int silently mismatches int-vs-str keys for list-typed
    submodules (e.g. jepa_predictor.predictor_blocks), a real bug hit and
    fixed in scripts/serve_qc_policy.py's _load_full_jepa_model; this avoids
    the whole class of issue by staying in nnx.State's own key space."""
    expected_flat = expected.flat_state()
    actual_flat = actual.flat_state()
    missing = [k for k in expected_flat if k not in actual_flat]
    if missing:
        raise ValueError(
            f"Restored state is missing {len(missing)} expected leaves (e.g. {missing[:5]}) -- "
            "checkpoint doesn't match the current model architecture."
        )
    pruned_flat = {k: actual_flat[k] for k in expected_flat}
    return nnx.State.from_flat_path(pruned_flat)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    merged = _merge_params(restored["train_state"], restored["params"])
    if state.ema_params is not None and merged.ema_params is not None:
        # Defense against checkpoints saved before the _split_params fix
        # above (e.g. stopgrad's step-29000 checkpoint): those have extra
        # non-Param buffer leaves (attn_mask) baked into the saved params
        # item, which come back with MORE structure than the abstract
        # state.ema_params template (built fresh via config.trainable_
        # filter, always Param-only) expects -- exactly what crashed pjit's
        # in_shardings structure check on --resume. Prune back down to
        # match state.ema_params regardless of what orbax actually
        # returned; for checkpoints saved by the fixed code this is a no-op
        # (structures already match).
        merged = dataclasses.replace(
            merged, ema_params=_prune_to_structure(merged.ema_params, state.ema_params)
        )
    return merged


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        # ema_params may only cover a trainable subset rather than the full
        # model (see jepa/train_step_transitions.py -- storing/blending
        # frozen leaves in EMA is pure waste since they never change).
        # Reconstruct the full tree here, at save time only, by filling in
        # whatever's missing from the live/online params. For a full-
        # structure ema_params (the common, non-JEPA case) the key sets
        # already match, so this is a no-op.
        #
        # Restrict the "full" side to nnx.Param leaves ONLY -- state.params
        # also contains non-Param nnx.Variable buffers (e.g. jepa/
        # ac_predictor_nnx.py's fixed causal attn_mask, or Dropout's RngKey/
        # RngCount state), which config.trainable_filter (nnx.All(nnx.Param,
        # ...)) never includes in ema_params in the first place. Filling gaps
        # from the UNFILTERED state.params blindly pulled those buffers in
        # too, giving a SAVED ema_params a different (larger) structure than
        # any FRESHLY-BUILT run's abstract ema_params shape (always Param-
        # only) -- silent until the next --resume, where pjit's in_shardings
        # structure check crashes on the mismatch (confirmed: this exact
        # failure on stopgrad's resume from step 29000, symmetric diff
        # {'attn_mask'} under ema_params['jepa_predictor']).
        full_params_only = state.params.filter(nnx.Param)
        ema_keys = set(params.flat_state())
        full_keys = set(full_params_only.flat_state())
        if ema_keys != full_keys:
            full_flat = dict(full_params_only.flat_state())
            full_flat.update(dict(params.flat_state()))
            params = nnx.State.from_flat_path(full_flat)
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
