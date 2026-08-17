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

import openpi.models.gemma as _gemma
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

# Pi0's PaliGemma.img(...) (pi0.py) builds SigLIP with num_classes=
# paligemma_config.width -- i.e. its output head already projects vision
# patch tokens into the Gemma LLM's embedding width (2048 for gemma_2b/
# gemma_2b_lora, NOT SigLIP's own internal width of 1152), so that image
# tokens and language tokens can be concatenated for the VLM prefix. That
# width does not match the 1408 the pretrained V-JEPA2-AC predictor's
# context input expects (that's ViT-g, V-JEPA2's own separate image
# encoder -- we've only converted its predictor, not its encoder).
# vision_proj bridges the two. This is only architecturally sound because
# JEPA and BC are co-trained end-to-end here: the projection (and the
# predictor itself) adapt to Pi0's actual feature distribution during
# training, rather than needing to already match a frozen predictor's
# pretraining distribution.
def _pi0_vision_width(model_config: _model.BaseModelConfig) -> int:
    return _gemma.get_config(model_config.paligemma_variant).width


class OpenPIWithJEPA(nnx.Module):
    """Wraps the base OpenPI model and registers the Action-Conditioned JEPA
    predictor as an NNX submodule, for single-step transition prediction."""

    def __init__(self, base_model: _model.BaseModel, config: _config.TrainConfig, rngs: nnx.Rngs):
        self.base_model = base_model

        # Defaults below match the pretrained V-JEPA2-AC ViT-g/16 predictor
        # checkpoint (see convert_checkpoint.py's --embed-dim/--num-frames/etc
        # CLI defaults) -- they must stay in sync with whatever architecture
        # convert_checkpoint.py was run with, or load_and_merge_predictor_state
        # will fail to map keys. TrainConfig has no fields for these; override
        # via getattr only if you add matching fields to TrainConfig.
        img_size = getattr(config, "img_size", (256, 256))
        patch_size = getattr(config, "patch_size", 16)
        # Single-step transitions -> single-frame context. If you later move
        # to true multi-frame video JEPA, these become >1 and the batch/data
        # loader need to supply frame histories instead of single transitions.
        num_frames = getattr(config, "num_frames", 8)
        tubelet_size = getattr(config, "tubelet_size", 2)
        embed_dim = getattr(config, "embed_dim", 1408)
        predictor_embed_dim = getattr(config, "predictor_embed_dim", 1024)
        # Action encoder's input dim -- must match the checkpoint's pretrained
        # action_encoder weight shape (7), not this project's own action_dim.
        action_dim = getattr(config, "action_dim", 7)

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
        self.vision_proj = nnx.Linear(_pi0_vision_width(config.model), embed_dim, use_bias=True, rngs=rngs)
        # Same bridging problem as vision_proj, for the predictor's action/
        # state encoders: they're pretrained with action_embed_dim=7 (the
        # V-JEPA2-AC checkpoint's own robot's action space), but Pi0 pads
        # actions/state to config.model.action_dim (32) for its own action
        # expert. kobo's native action space (see KoboOutputs) doesn't match
        # V-JEPA2's pretraining robot either, so this is a genuine new space,
        # not just a reshape -- relying on end-to-end fine-tuning to adapt
        # it, same as vision_proj.
        self.action_proj = nnx.Linear(config.model.action_dim, action_dim, use_bias=True, rngs=rngs)
        self.state_proj = nnx.Linear(config.model.action_dim, action_dim, use_bias=True, rngs=rngs)

        # See TrainConfig.jepa_stopgrad_vision's docstring (config.py) -- an
        # nnx.Module attribute rather than a getattr(config, ...) call at use
        # site (unlike the other JEPA hyperparams) because extract_vision_latents
        # needs it and doesn't otherwise have config in scope.
        self.jepa_stopgrad_vision = getattr(config, "jepa_stopgrad_vision", False)

    def __call__(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def compute_loss(self, *args, **kwargs):
        return self.base_model.compute_loss(*args, **kwargs)

    def extract_vision_latents(self, obs: _model.Observation) -> jnp.ndarray:
        """Runs Pi0's own SigLIP tower (the same PaliGemma.img(...) call
        embed_prefix uses for the BC prefix, see pi0.py) on the primary
        exterior camera and projects its patch tokens into the JEPA
        predictor's embedding space via vision_proj. Only the "base_0_rgb"
        camera is used -- concatenating multiple cameras here would inflate
        the token count in a way the predictor's forward() would silently
        misinterpret as multiple time frames (T = N_ctxt // (grid_h*grid_w)),
        not multiple camera views.

        If jepa_stopgrad_vision is set, image_tokens (PaliGemma.img's raw
        output, before vision_proj) is stop-gradiented here -- this is the
        ONE place both the online JEPA-context call (train_step's z_context)
        and the teacher call (h_raw, already fully detached anyway via
        _get_teacher_model's stop-gradiented params) go through, so gating it
        here cuts JEPA's gradient contribution to the SHARED PaliGemma/SigLIP
        weights (which BC's own embed_prefix call also depends on for action
        prediction) without touching jepa_predictor/vision_proj/action_proj/
        state_proj's own training -- those still get real gradients, just
        computed on top of a frozen (for JEPA's purposes) vision feature.
        Inference-only callers (compute_intrinsic_reward, qc_label_rewards.py,
        inference_online.py) never differentiate through this at all, so the
        stop_gradient is a no-op for them either way.
        """
        image_tokens, _ = self.base_model.PaliGemma.img(obs.images["base_0_rgb"], train=False)
        if self.jepa_stopgrad_vision:
            image_tokens = jax.lax.stop_gradient(image_tokens)
        return self.vision_proj(image_tokens)


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
    if state.ema_params is not None:
        # state.ema_params only covers config.trainable_filter's leaves (see
        # init_train_state) -- frozen leaves never change during training, so
        # storing/blending a second full-size copy of them would be pure
        # waste. Merge the EMA slice back over the always-current frozen
        # leaves from state.params to get a complete param tree.
        full_flat = dict(state.params.flat_state())
        full_flat.update(dict(state.ema_params.flat_state()))
        teacher_params = nnx.State.from_flat_path(full_flat)
    else:
        teacher_params = state.params
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
        action_t = model_inst.action_proj(action_chunk[:, 0, :])   # [B, action_dim] -> [B, 7]
        proprio_t = model_inst.state_proj(_get_proprio(obs_t))    # [B, action_dim] -> [B, 7]

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
        beta_init = getattr(config, "beta_jepa", 0.5)
        beta_final = getattr(config, "beta_jepa_final", None)
        if beta_final is not None:
            # Cosine decay beta_init -> beta_final over beta_jepa_decay_steps,
            # then hold at beta_final. state.step is a traced value (this
            # jitted train_step is compiled once for the whole run), so the
            # schedule has to be computed with jnp ops here rather than as a
            # plain Python float baked in at trace time -- see TrainConfig.
            # beta_jepa_final's docstring for the motivation.
            decay_steps = getattr(config, "beta_jepa_decay_steps", None) or config.num_train_steps
            progress = jnp.clip(state.step / decay_steps, 0.0, 1.0)
            cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
            beta = beta_final + (beta_init - beta_final) * cosine
        else:
            # Cast explicitly: unlike the decay branch (whose jnp ops already
            # produce a tracer), a constant beta_init is a bare Python float
            # here, which fails train_step's `-> dict[str, at.Array]` return
            # annotation once it's put straight into the metrics dict below
            # (the multiplication into total_loss happened to upcast it
            # implicitly, masking this for that value alone).
            beta = jnp.asarray(beta_init)
        total_loss = alpha * l_bc_mean + beta * l_jepa

        return total_loss, {
            "loss": total_loss,
            "loss_bc": l_bc_mean,
            "loss_jepa": l_jepa,
            "beta_jepa": beta,
        }

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, obs_t, action_chunk, obs_t1
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_full_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_full_params, opt_state=new_opt_state)

    if state.ema_decay is not None and state.ema_params is not None:
        # state.ema_params only covers config.trainable_filter's leaves (see
        # init_train_state) -- frozen leaves (the multi-GB Gemma/SigLIP
        # backbone under LoRA finetuning) never change from nnx.update()
        # above, so tracking/blending a second full-size copy of them would
        # be pure waste. That's what OOM'd this LoRA "low_mem" config the
        # first time EMA was turned on: a full-structure ema_params doubles
        # resident frozen-param memory, and since it's a separate top-level
        # jit output from state.params, XLA can't just alias the two even
        # when the blend is skipped for those leaves -- it has to physically
        # duplicate the buffer. Keeping ema_params sparse avoids that
        # entirely; _get_teacher_model() merges it back over state.params.
        #
        # nnx.State.map()/jax.tree.map() both operate through the
        # VariableState wrapper at each leaf (see nnx_utils.state_map's
        # `p.replace(...)` pattern), so build the blended state via
        # flat_state() dicts directly rather than a tree_map over a
        # separately-built boolean mask (which has a mismatched pytree
        # structure -- raw bools where a VariableState node is expected).
        new_trainable = new_full_params.filter(config.trainable_filter)
        old_flat = dict(state.ema_params.flat_state())
        new_flat = dict(new_trainable.flat_state())

        blended_flat = {}
        for k, new_vs in new_flat.items():
            # Non-floating leaves (e.g. nnx.Rngs' RngKey/RngCount state, which
            # rides along in nnx.state(model) alongside the real Params) can't
            # be blended -- decay*old + (1-decay)*new is undefined for PRNG
            # keys/counters, and there's no meaningful "average" of two of
            # them anyway. Just track the live online value for those.
            if jnp.issubdtype(jnp.asarray(new_vs.value).dtype, jnp.floating):
                old_vs = old_flat[k]
                blended_flat[k] = new_vs.replace(
                    state.ema_decay * old_vs.value + (1.0 - state.ema_decay) * new_vs.value
                )
            else:
                blended_flat[k] = new_vs

        new_state = dataclasses.replace(
            new_state,
            ema_params=nnx.State.from_flat_path(blended_flat),
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
    action_t: jnp.ndarray,   # [B, model.config.action_dim] -- single-step action (Pi0's padded
                             # dim, e.g. 32), NOT a chunk -- same convention as action_chunk[:, 0, :]
                             # in train_step's loss_fn.
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
    # action_t/proprio arrive in Pi0's padded action_dim (e.g. 32); the
    # predictor's action/state encoders are pretrained on the V-JEPA2-AC
    # checkpoint's native 7-dim space (see convert_checkpoint.py) -- train_step's
    # loss_fn applies these same projections before calling jepa_predictor, and
    # skipping them here (as the original version of this function did) is a
    # shape mismatch against action_encoder's (7, 1024) kernel, since this was
    # never actually exercised by any caller before now.
    action_t = model.action_proj(action_t)
    proprio_t = model.state_proj(_get_proprio(obs_t))
    z_context = model.extract_vision_latents(obs_t)
    z_pred = model.jepa_predictor(
        z_context,
        actions=action_t[:, None, :],
        states=proprio_t[:, None, :],
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