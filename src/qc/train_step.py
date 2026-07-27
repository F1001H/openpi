"""Chunked TD critic training step for Q-chunking (Li/Zhou/Levine, NeurIPS
2025, https://github.com/ColinQiyangLi/qc), ported from the reference's
ACFQLAgent.critic_loss (agents/acfql.py:22-52) to NNX.

No actor NETWORK exists here (best-of-N candidate generation reuses Pi0.5's
own flow-matching action expert directly, src/qc/actor.py -- used for live
inference-time action selection, Phase 2b). Critic training itself still
never needs the VLA model loaded: the TD target's "next actions" are
best-of-N-scored from a set of candidate action chunks that were sampled
from the FROZEN BC actor and cached once at labeling time (see
scripts/qc_label_rewards.py), not sampled live during training -- since the
BC actor doesn't change during critic training, this is equivalent to live
sampling but keeps the same VLA-model decoupling Phase 1 established for
reward/embeddings. See src/utils/data_loader.py's QChunkTransitionDataset for
how these are read back out of the cache.
"""

import flax.nnx as nnx
from flax import struct
import jax
import jax.numpy as jnp
import optax

from qc.critic import QChunkCritic


@struct.dataclass
class QCTrainState:
    step: jax.Array
    params: nnx.State
    target_params: nnx.State
    model_def: nnx.GraphDef
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False)
    horizon_length: int = struct.field(pytree_node=False)
    q_agg: str = struct.field(pytree_node=False)


def init_qc_train_state(
    rng: jax.Array,
    embed_dim: int,
    proprio_dim: int,
    action_dim: int,
    horizon_length: int,
    *,
    lr: float = 3e-4,
    tau: float = 0.005,
    discount: float = 0.99,
    q_agg: str = "mean",
    num_qs: int = 2,
    hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
    layer_norm: bool = True,
) -> QCTrainState:
    critic = QChunkCritic(
        embed_dim, proprio_dim, action_dim, horizon_length,
        hidden_dims=hidden_dims, num_qs=num_qs, layer_norm=layer_norm, rngs=nnx.Rngs(rng),
    )
    graphdef, params = nnx.split(critic)
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)
    return QCTrainState(
        step=jnp.array(0),
        params=params,
        # Target starts equal to online, matching the reference's
        # `params['modules_target_critic'] = params['modules_critic']` (acfql.py:307).
        target_params=params,
        model_def=graphdef,
        opt_state=opt_state,
        tx=tx,
        tau=tau,
        discount=discount,
        horizon_length=horizon_length,
        q_agg=q_agg,
    )


Batch = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def critic_loss_fn(state: QCTrainState, critic_params: nnx.State, batch: Batch) -> tuple[jnp.ndarray, dict]:
    embed_t, proprio_t, action_chunk, reward, embed_th, proprio_th, next_action_candidates, mask = batch
    # next_action_candidates: [B, num_candidates, horizon_length, action_dim]
    batch_size, num_candidates = next_action_candidates.shape[:2]

    critic = nnx.merge(state.model_def, critic_params)
    target_critic = nnx.merge(state.model_def, jax.lax.stop_gradient(state.target_params))

    # Score every candidate per example with the target critic, aggregate
    # over the ensemble (q_agg), THEN take the best-scoring candidate per
    # example -- matches the reference's best-of-N target selection order
    # (aggregate ensemble first, then argmax over candidates; acfql.py's
    # sample_actions best-of-n branch, acfql.py:180-204).
    embed_th_tiled = jnp.repeat(embed_th, num_candidates, axis=0)
    proprio_th_tiled = jnp.repeat(proprio_th, num_candidates, axis=0)
    candidates_flat = next_action_candidates.reshape(batch_size * num_candidates, *next_action_candidates.shape[2:])

    next_qs = target_critic(embed_th_tiled, proprio_th_tiled, candidates_flat)  # [num_qs, B*num_candidates]
    next_q_per_cand = next_qs.min(axis=0) if state.q_agg == "min" else next_qs.mean(axis=0)
    next_q_per_cand = next_q_per_cand.reshape(batch_size, num_candidates)
    next_q = next_q_per_cand.max(axis=1)  # [B] -- best candidate per example

    # n-step chunked Bellman target: discount**horizon_length (the whole
    # chunk is one macro-action), matching acfql.py:40-41.
    target_q = reward + (state.discount**state.horizon_length) * mask * jax.lax.stop_gradient(next_q)

    qs = critic(embed_t, proprio_t, action_chunk)  # [num_qs, B]
    critic_loss = jnp.mean(((qs - target_q[None, :]) ** 2) * mask[None, :])

    return critic_loss, {
        "critic_loss": critic_loss,
        "q_mean": qs.mean(),
        "q_max": qs.max(),
        "q_min": qs.min(),
        "q_std": qs.std(),
        "target_q_mean": target_q.mean(),
    }


@jax.jit
def train_step(state: QCTrainState, batch: Batch) -> tuple[QCTrainState, dict]:
    (loss, info), grads = jax.value_and_grad(
        lambda p: critic_loss_fn(state, p, batch), has_aux=True
    )(state.params)

    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # Target network EMA: new_target = tau*online + (1-tau)*old_target,
    # matching the reference's target_update (acfql.py:129-136).
    new_target_params = jax.tree.map(
        lambda new, old: state.tau * new + (1 - state.tau) * old,
        new_params, state.target_params,
    )

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        target_params=new_target_params,
        opt_state=new_opt_state,
    )
    return new_state, info
