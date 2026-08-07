"""Best-of-N critic-scored action sampling for Q-chunking, replacing Phase
1's SARSA-style TD target with the reference algorithm's real off-policy
target (ColinQiyangLi/qc's `agents/acfql.py`, `actor_type="best-of-n"` path,
acfql.py:180-204: sample N candidate chunks from the BC actor, score with the
critic, keep the best).

Operates on a full training BATCH of B observations at once: for each of the
B observations independently, samples num_samples candidates (B*num_samples
total, one batched flow-matching call, not B*num_samples separate ones) and
returns the best-scoring candidate per observation.
"""

import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.transforms as _transforms

from qc.critic import QChunkCritic


def best_of_n_action_batch(
    rng: jax.Array,
    base_model,  # OpenPIWithJEPA (src/jepa/train_step_transitions.py) -- needs
                 # .base_model.sample_actions and .extract_vision_latents
    critic: QChunkCritic,
    obs: _model.Observation,  # batch dim B
    proprio: jnp.ndarray,  # [B, proprio_dim]
    num_samples: int,
    horizon_length: int,
    action_dim: int,
    norm_stats: dict,  # data_config.norm_stats -- must include an "actions" entry
    *,
    use_quantile_norm: bool = False,
    num_flow_steps: int = 10,
    q_agg: str = "mean",
) -> jnp.ndarray:
    """Returns [B, horizon_length, action_dim] -- the best of num_samples
    candidates per observation, per the critic's scoring."""
    batch_size = proprio.shape[0]

    # [B] -> [B*num_samples], grouped so reshape(B, num_samples, ...) recovers
    # per-observation candidate groups (jnp.repeat, not jnp.tile).
    tiled_obs = jax.tree.map(lambda x: jnp.repeat(x, num_samples, axis=0), obs)
    candidates = base_model.base_model.sample_actions(rng, tiled_obs, num_steps=num_flow_steps)
    # candidates: [B*num_samples, action_horizon, model_action_dim], in the
    # NORMALIZED units the model was trained on (z-scored via config.data's
    # Normalize transform) -- must be unnormalized back to native physical
    # units BEFORE slicing to the critic's native action_dim, matching
    # Policy's real output pipeline order (Unnormalize happens before
    # KoboOutputs' slice, see policy_config.create_trained_policy). Skipping
    # this would silently feed wrong-scale actions to the critic (mixed with
    # the OTHER action input, action_chunk, which comes from the raw dataset
    # in native units) -- caught during Phase 2b review, not Phase 2a testing.
    unnormalize = _transforms.Unnormalize(norm_stats, use_quantiles=use_quantile_norm)
    candidates = unnormalize({"actions": candidates})["actions"]
    candidates = candidates[:, :horizon_length, :action_dim]  # [B*N, h, a]

    embed = jnp.mean(base_model.extract_vision_latents(obs), axis=1)  # [B, D] (context, computed once per obs)
    embed_tiled = jnp.repeat(embed, num_samples, axis=0)  # [B*N, D]
    proprio_tiled = jnp.repeat(proprio, num_samples, axis=0)  # [B*N, proprio_dim]

    qs = critic(embed_tiled, proprio_tiled, candidates)  # [num_qs, B*N]
    q = qs.min(axis=0) if q_agg == "min" else qs.mean(axis=0)  # [B*N]
    q = q.reshape(batch_size, num_samples)
    best_idx = jnp.argmax(q, axis=1)  # [B]

    candidates = candidates.reshape(batch_size, num_samples, horizon_length, action_dim)
    return candidates[jnp.arange(batch_size), best_idx]  # [B, h, a]
