#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""
CHANGES FROM THE ORIGINAL SCRIPT (see prior review for full context):

1. Batch is now a (obs_t, action_chunk, obs_t1) triple, not (obs, actions).
   obs_t1 is the observation one env-step after obs_t. This requires your
   data loader / dataset to actually produce shifted next-observations --
   that change isn't included here since I don't have _data_loader.py.

2. ASSUMPTION TO VERIFY: `action_t = action_chunk[:, 0, :]` treats the FIRST
   action in the BC action-chunk as "the action that produced the obs_t ->
   obs_t1 transition." This is only correct if your chunk's first entry is a
   single physical env step and obs_t1 is exactly one step later. If your
   action horizon groups multiple env steps per chunk differently, adjust
   this indexing (and possibly which obs you use as obs_t1) accordingly.

3. JEPA target is now genuinely computed from obs_t1 through the EMA teacher
   (was: computed from obs_t, same as context -- a trivial, uninformative
   target).

4. `target_norm` is now applied via `target_model` (the EMA-merged instance,
   already fully stop-gradiented) instead of via `model_inst`. Previously the
   norm's params were the *online* model's, trainable, sitting on the target
   side of an L2/L1 loss -- letting the model shrink the loss by warping the
   target rather than improving the prediction. This is the representation-
   collapse risk flagged earlier.

5. `state.ema_params` is no longer accessed unconditionally -- if
   `config.ema_decay is None` (no EMA configured), the teacher falls back to
   `state.params` itself (still stop_gradient'd, just not a running average).
   Not "true" EMA-JEPA in that case, but it no longer crashes.

6. `obs.get("state", acts)` (invalid on a dataclass, wrong fallback shape) is
   replaced with an explicit `getattr(obs, "state", None)`. If your
   Observation type names this field differently, change this one line.

7. Added `compute_intrinsic_reward`: a separate, non-training function meant
   to be called at rollout/inference time to turn prediction error into a
   scalar-per-transition curiosity reward, decoupled from the BC gradient step.
"""

import dataclasses
import functools
import logging
import platform
from typing import Any, Dict, Tuple

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
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

# Ported NNX predictor (was: torch jepa.ac_predictor.VisionTransformerPredictorAC)
from jepa.ac_predictor_nnx import VisionTransformerPredictorAC


# =========================================================================== #
# 1. MODEL WRAPPER
# =========================================================================== #

class OpenPIWithJEPA(nnx.Module):
    """Wraps the base OpenPI model and registers the Action-Conditioned JEPA
    predictor as an NNX submodule, for single-step transition prediction."""

    def __init__(self, base_model: _model.BaseModel, config: _config.TrainConfig, rngs: nnx.Rngs):
        self.base_model = base_model

        img_size = getattr(config, "img_size", (224, 224))
        patch_size = getattr(config, "patch_size", 16)
        # Single-step transitions -> single-frame context. If you later move
        # to true multi-frame video JEPA, these become >1 and the batch/data
        # loader need to supply frame histories instead of single transitions.
        num_frames = getattr(config, "num_frames", 1)
        tubelet_size = getattr(config, "tubelet_size", 1)
        embed_dim = getattr(config, "embed_dim", 768)
        predictor_embed_dim = getattr(config, "predictor_embed_dim", 1024)
        action_dim = getattr(config, "action_dim", 14)

        self.jepa_predictor = VisionTransformerPredictorAC(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            action_embed_dim=action_dim,
            rngs=rngs,
        )
        self.target_norm = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def compute_loss(self, *args, **kwargs):
        return self.base_model.compute_loss(*args, **kwargs)

    def extract_vision_latents(self, obs: _model.Observation) -> jnp.ndarray:
        if hasattr(self.base_model, "extract_vision_latents"):
            return self.base_model.extract_vision_latents(obs)
        elif hasattr(self.base_model, "backbone") and hasattr(self.base_model.backbone, "extract_features"):
            return self.base_model.backbone.extract_features(obs)
        else:
            raise AttributeError(
                "Base model structure does not expose a recognized method for visual latent token extraction."
            )


def _get_proprio(obs: _model.Observation) -> jnp.ndarray:
    """Pull the proprioceptive state off an Observation. Adjust the attribute
    name here if your Observation type calls it something other than `state`."""
    state = getattr(obs, "state", None)
    if state is None:
        raise AttributeError(
            "Observation has no `.state` attribute -- update _get_proprio() with the correct field name."
        )
    return state


def _get_teacher_model(state: training_utils.TrainState):
    """Return a fully stop-gradiented model to use as the JEPA target
    network. Falls back to the online params if no EMA is configured."""
    teacher_params = state.ema_params if state.ema_params is not None else state.params
    return nnx.merge(state.model_def, jax.lax.stop_gradient(teacher_params))


# =========================================================================== #
# 2. TRAIN STEP (transition-based)
# =========================================================================== #

@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, _model.Observation],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Action-Conditioned JEPA + OpenPI BC train step over (obs_t, actions, obs_t1) transitions."""

    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)
    obs_t, action_chunk, obs_t1 = batch

    @at.typecheck
    def loss_fn(
        model_inst: OpenPIWithJEPA,
        step_rng: at.KeyArrayLike,
        obs_t: _model.Observation,
        action_chunk: _model.Actions,
        obs_t1: _model.Observation,
    ):
        vla_rng, jepa_rng = jax.random.split(step_rng)

        # --- Task 1: Policy BC loss (unchanged, uses the full action chunk) ---
        l_bc = model_inst.compute_loss(vla_rng, obs_t, action_chunk, train=True)
        l_bc_mean = jnp.mean(l_bc)

        # --- Task 2: Action-conditioned JEPA transition prediction ---
        # ASSUMPTION (see module docstring point 2): first action in the chunk
        # is the single physical action that produced obs_t -> obs_t1.
        action_t = action_chunk[:, 0, :]                    # [B, action_dim]
        proprio_t = _get_proprio(obs_t)                     # [B, action_dim] (or whatever your state dim is)

        z_context = model_inst.extract_vision_latents(obs_t)  # [B, H*W, D] for num_frames=1

        # predictor expects [B, T, action_dim] with T=1 here
        z_pred = model_inst.jepa_predictor(
            z_context,
            actions=action_t[:, None, :],
            states=proprio_t[:, None, :],
        )

        # --- Task 3: Momentum target from the NEXT observation ---
        target_model = _get_teacher_model(state)
        h_raw = target_model.extract_vision_latents(obs_t1)
        # Apply the norm through the *teacher* copy -- stop-gradient already
        # covers all of target_model's params, so this can't be gamed by
        # training target_norm to chase z_pred.
        h = target_model.target_norm(h_raw)

        # --- Task 4: Combined objective ---
        loss_exp = getattr(config, "jepa_loss_exp", 2.0)
        error_jepa = jnp.abs(z_pred - h) ** loss_exp
        l_jepa = jnp.mean(error_jepa) / loss_exp

        alpha = getattr(config, "alpha_bc", 1.0)
        beta = getattr(config, "beta_jepa", 0.5)
        total_loss = alpha * l_bc_mean + beta * l_jepa

        return total_loss, {
            "loss": total_loss,
            "loss_bc": l_bc_mean,
            "loss_jepa": l_jepa,
        }

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads, metrics = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, obs_t, action_chunk, obs_t1
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_full_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_full_params, opt_state=new_opt_state)

    if state.ema_decay is not None and state.ema_params is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1.0 - state.ema_decay) * new,
                state.ema_params,
                new_full_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    info = {
        **metrics,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


# =========================================================================== #
# 3. INTRINSIC REWARD (inference-only, for use during rollout / RL)
# =========================================================================== #

def compute_intrinsic_reward(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    obs_t: _model.Observation,
    action_t: jnp.ndarray,   # [B, action_dim] -- single-step action, NOT a chunk
    obs_t1: _model.Observation,
) -> jnp.ndarray:
    """Per-transition curiosity reward from JEPA prediction error.
    Returns shape [B] -- reduces over tokens/features but NOT over batch,
    unlike the training loss. No gradients are computed here; this is meant
    to be called from your rollout/collection loop, not from train_step.

    NOTE: uses the online model for context+prediction and the same teacher
    (EMA or online, per _get_teacher_model) for the target, matching
    train_step's convention. If you want reward computed purely from a fixed
    snapshot of the policy (e.g. to keep it stable across a rollout), pass in
    a `state` you've deliberately frozen rather than the live training state.
    """
    model = nnx.merge(state.model_def, jax.lax.stop_gradient(state.params))
    z_context = model.extract_vision_latents(obs_t)
    z_pred = model.jepa_predictor(
        z_context,
        actions=action_t[:, None, :],
        states=_get_proprio(obs_t)[:, None, :],
    )

    target_model = _get_teacher_model(state)
    h_raw = target_model.extract_vision_latents(obs_t1)
    h = target_model.target_norm(h_raw)

    loss_exp = getattr(config, "jepa_loss_exp", 2.0)
    error = jnp.abs(z_pred - h) ** loss_exp
    # reduce over all axes except batch (axis 0)
    reduce_axes = tuple(range(1, error.ndim))
    reward = jnp.mean(error, axis=reduce_axes) / loss_exp
    return reward  # [B]