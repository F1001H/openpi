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
    uncertainty_penalty: float = 0.0,
    actor_disagreement_penalty: float = 0.0,
    critic_weight: float = 1.0,
    maximize_score: bool = False,
    selection_mode: str = "score",
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
    candidates_grouped = candidates.reshape(batch_size, num_samples, horizon_length, action_dim)

    # Actor-side uncertainty (per your PI's suggestion): how much do the
    # num_samples candidates disagree with EACH OTHER at this observation,
    # independent of anything the critic says. Distinct from the critic's
    # cross-head `disagreement` below -- that flags candidates the critic is
    # unsure how to SCORE; this flags candidates that are themselves outliers
    # relative to what the actor mostly proposes here, i.e. the actor's own
    # sampling distribution was less consistent/confident at this state.
    # Free to compute -- no extra forward pass, just stats over samples we
    # already drew.
    action_mean = jnp.mean(candidates_grouped, axis=1, keepdims=True)  # [B, 1, h, a]
    actor_disagreement = jnp.sqrt(jnp.mean((candidates_grouped - action_mean) ** 2, axis=(2, 3)))  # [B, N]

    embed = jnp.mean(base_model.extract_vision_latents(obs), axis=1)  # [B, D] (context, computed once per obs)
    embed_tiled = jnp.repeat(embed, num_samples, axis=0)  # [B*N, D]
    proprio_tiled = jnp.repeat(proprio, num_samples, axis=0)  # [B*N, proprio_dim]

    qs = critic(embed_tiled, proprio_tiled, candidates)  # [num_qs, B*N]
    q = qs.min(axis=0) if q_agg == "min" else qs.mean(axis=0)  # [B*N]
    # Cross-head disagreement (critic.py already trains num_qs>=2 heads for
    # the usual conservative-Q-learning reason, so this is free -- no extra
    # forward pass). A proxy for the critic's own epistemic uncertainty
    # about this candidate, distinct from q itself (which conflates genuine
    # low predicted error with the critic just being confidently wrong).
    disagreement = qs.std(axis=0)  # [B*N]
    # score = predicted error + penalty * disagreement, minimized below --
    # NOT a reward to maximize. This is the pessimistic-under-uncertainty
    # direction for a MINIMIZATION objective: a candidate the heads disagree
    # on gets treated as riskier (effectively higher assumed error) than its
    # raw q alone suggests, on top of (not instead of) preferring low q.
    # uncertainty_penalty=0.0 (default) reduces to plain argmin(q).
    # actor_disagreement_penalty stacks the actor-side term on top of the
    # critic-side ones -- all three are independently weighted, so any
    # subset can be zeroed out to isolate its effect. critic_weight scales
    # the critic's own q term specifically -- default 1.0 (unchanged
    # behavior); critic_weight=0.0 drops the critic out of selection
    # entirely, isolating what --actor-disagreement-penalty alone does (was
    # the critic's q contributing anything on top of "prefer the most
    # consensus/least-outlier candidate the actor itself proposed", or was
    # the actor-disagreement term already doing all the work in the
    # argmax+actor-disagree result that beat plain BC?).
    score = (
        critic_weight * q.reshape(batch_size, num_samples)
        + uncertainty_penalty * disagreement.reshape(batch_size, num_samples)
        + actor_disagreement_penalty * actor_disagreement
    )
    # argmin by default, not argmax: the critic predicts future JEPA
    # prediction ERROR (src/jepa/train_step_transitions.py's
    # compute_intrinsic_reward), a curiosity/novelty signal -- maximizing it
    # picks the candidate expected to be LEAST predictable, which is exactly
    # the wrong criterion for a frozen actor+critic at eval time (no further
    # learning happens from this episode, so there's no payoff to "exploring"
    # a risky candidate, only downside). Minimizing it instead picks the
    # candidate closest to what the JEPA world model confidently expects --
    # a proxy for "looks like the in-distribution, demo-like continuation
    # the BC actor was trained on," which should correlate with success far
    # better than raw novelty does. Empirically motivated by qc_full_
    # finetune_beta0.1_step29999's suite results: most best-of-N failures
    # clustered at the grasp/contact phase specifically -- the single
    # hardest moment for any vision-based world model to predict, so argmax
    # was actively steering toward messier, less confident contacts.
    # maximize_score=True restores the original argmax behavior -- exposed
    # as a flag (not just deleted) so this can be picked empirically per the
    # argmin-vs-argmax A/B result on real eval data, rather than assumed.
    if selection_mode == "majority_vote":
        # Each of the num_qs heads votes independently for its OWN preferred
        # candidate (argmin/argmax over its own raw Q, ignoring q_agg/
        # uncertainty_penalty/actor_disagreement_penalty -- those are ways of
        # combining heads into one score, which is exactly what voting
        # replaces with a committee decision instead). The candidate with
        # the most votes wins -- plurality, not a guaranteed true majority
        # (with num_qs=5 candidates split 1-1-1-1-1 across 5 different
        # picks, this just takes the first one encountered; a real majority,
        # >=3/5 agreeing, isn't guaranteed to exist). Motivated as a more
        # principled alternative to uncertainty_penalty's "penalize
        # disagreement" approach: instead of guessing how much to penalize
        # cross-head disagreement, let the heads settle it by committee.
        qs_grouped = qs.reshape(-1, batch_size, num_samples)  # [num_qs, B, N]
        per_head_best = (
            jnp.argmax(qs_grouped, axis=-1) if maximize_score else jnp.argmin(qs_grouped, axis=-1)
        )  # [num_qs, B]
        votes = jax.nn.one_hot(per_head_best, num_samples)  # [num_qs, B, N]
        vote_counts = votes.sum(axis=0)  # [B, N]
        best_idx = jnp.argmax(vote_counts, axis=-1)  # [B]
    else:
        best_idx = jnp.argmax(score, axis=1) if maximize_score else jnp.argmin(score, axis=1)  # [B]

    return candidates_grouped[jnp.arange(batch_size), best_idx]  # [B, h, a]
