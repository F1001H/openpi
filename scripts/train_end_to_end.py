#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import dataclasses
import functools
import logging
import platform
import math
from typing import Any, Dict, Tuple

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
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

# Explicit import linking to your custom ported JAX/Flax Action-Conditioned Predictor module
from jepa.ac_predictor import VisionTransformerPredictorAC


# =========================================================================== #
# 1. DYNAMIC MODEL WRAPPER FOR JEPA INTEGRATION
# =========================================================================== #

class OpenPIWithJEPA(nnx.Module):
    """
    Dynamic wrapper that intercepts the base OpenPI model to register the 
    Action-Conditioned JEPA modules into the Flax NNX state tracking tree.
    """
    def __init__(self, base_model: _model.BaseModel, config: _config.TrainConfig, rngs: nnx.Rngs):
        self.base_model = base_model
        
        # Pull configuration parameters safely, fallback to default dimensions if not explicitly set
        img_size = getattr(config, "img_size", (224, 224))
        patch_size = getattr(config, "patch_size", 16)
        num_frames = getattr(config, "num_frames", 4)
        tubelet_size = getattr(config, "tubelet_size", 1)
        embed_dim = getattr(config, "embed_dim", 768)
        predictor_embed_dim = getattr(config, "predictor_embed_dim", 1024)
        action_dim = getattr(config, "action_dim", 14) # Default to bimanual footprint (e.g. 2x 7DoF)

        # Bind the Action-Conditioned Predictor to the active NNX tree
        self.jepa_predictor = VisionTransformerPredictorAC(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            action_embed_dim=action_dim,
            rngs=rngs
        )
        
        # Bind the target normalization layer
        self.target_norm = nnx.LayerNorm(embed_dim)

    def __call__(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def compute_loss(self, *args, **kwargs):
        return self.base_model.compute_loss(*args, **kwargs)

    def extract_vision_latents(self, obs: _model.Observation) -> jnp.ndarray:
        """
        Extract visual token features from the base VLA policy model representation space.
        """
        # Checks if your base model class exposes an explicit method, otherwise fall back to common hooks
        if hasattr(self.base_model, "extract_vision_latents"):
            return self.base_model.extract_vision_latents(obs)
        elif hasattr(self.base_model, "backbone") and hasattr(self.base_model.backbone, "extract_features"):
            return self.base_model.backbone.extract_features(obs)
        else:
            raise AttributeError("Base model structure does not expose a recognized method for visual latent token extraction.")


# =========================================================================== #
# 2. CORE UNIFIED TRAINING OPTIMIZATION LOOP STEP
# =========================================================================== #

@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Action-Conditioned V-JEPA Multi-Objective Train Step tailored for OpenPI."""
    
    # Unpack model blueprint definitions and map dynamic graph arrays
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    @at.typecheck
    def loss_fn(
        model_inst: OpenPIWithJEPA, 
        step_rng: at.KeyArrayLike, 
        obs: _model.Observation, 
        acts: _model.Actions
    ):
        vla_rng, jepa_rng = jax.random.split(step_rng)
        
        # --- Task 1: Policy Behavioral Cloning (BC) Core Loss ---
        l_bc = model_inst.compute_loss(vla_rng, obs, acts, train=True)
        l_bc_mean = jnp.mean(l_bc)

        # --- Task 2: Action-Conditioned JEPA State-Space Transitions ---
        # Extract visual feature tracks directly out of the active encoder backbone representation space
        z_context = model_inst.extract_vision_latents(obs)
        
        # Safely capture proprioceptive feedback or default to direct kinematic representations
        proprio_states = obs.get("state", acts)
        
        # Feed token dynamics directly into our Action-Conditioned Predictor structure
        z_pred = model_inst.jepa_predictor(z_context, acts, proprio_states)
        
        # --- Task 3: Momentum Target Evaluation (EMA Path) ---
        # Safeguard structural weights against optimization collapse using an explicit stop_gradient boundary
        target_model = nnx.merge(state.model_def, jax.lax.stop_gradient(state.ema_params))
        h_raw = target_model.extract_vision_latents(obs)
        
        # Apply normalization transformation to balance multi-layer distillation target variances
        h = model_inst.target_norm(h_raw)

        # --- Task 4: Combined Optimization Objective Resolution ---
        loss_exp = getattr(config, "jepa_loss_exp", 2.0)
        error_jepa = jnp.abs(z_pred - h) ** loss_exp
        l_jepa = jnp.mean(error_jepa) / loss_exp

        # Dynamic parameter scalar unpacking
        alpha = getattr(config, "alpha_bc", 1.0)
        beta = getattr(config, "beta_jepa", 0.5)
        total_loss = alpha * l_bc_mean + beta * l_jepa

        return total_loss, {
            "loss": total_loss,
            "loss_bc": l_bc_mean,
            "loss_jepa": l_jepa
        }

    # Restrict gradient computations exclusively to params matching config trainable parameters maps
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads, metrics = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # In-place synchronization updating across tracking graphs
    nnx.update(model, new_params)
    new_full_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_full_params, opt_state=new_opt_state)
    
    # --- Task 5: Momentum Weight Target Parameter Updates ---
    if state.ema_decay is not None and state.ema_params is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1.0 - state.ema_decay) * new, 
                state.ema_params, 
                new_full_params
            ),
        )

    # Filter out active weights tracking matrix for logging diagnostics
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
# 3. RUNTIME TELEMETRY AND WEIGHT INITIALIZATION HOOKS
# =========================================================================== #

def init_logging():
    """Custom logging formatting block."""
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
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        
        # 1. Instantiate native OpenPI Base Model
        base_model = config.model.create(model_rng)
        
        # 2. Intercept and wrap with the active Action-Conditioned JEPA modules
        rng, jepa_rng = jax.random.split(rng)
        model = OpenPIWithJEPA(base_model, config, nnx.Rngs(jepa_rng))

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

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
# 4. MAIN TRAINING DEPLOYMENT EXECUTIVE INTERFACE
# =========================================================================== #

def main(config: _config.TrainConfig):
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

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader structure:\n{training_utils.array_tree_to_info(batch)}")

    # Log initial visual sample images to verification boards
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state parameters footprint:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

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
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Terminating process iterations. Flushing checkpoint queues...")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())