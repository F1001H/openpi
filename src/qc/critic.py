"""NNX critic ensemble for Q-chunking (Li/Zhou/Levine, NeurIPS 2025,
https://github.com/ColinQiyangLi/qc).

Consumes the SAME observation representation the Q-chunking cache produces
(scripts/qc_label_rewards.py): a pooled JEPA vision embedding + raw low-dim
proprio state, concatenated with a flattened action chunk. This is a small
MLP -- it does NOT touch the VLA/JEPA backbone at all, matching the design
in src/utils/data_loader.py's QChunkTransitionDataset (see that file's
module docstring for the full rationale).

Architecture/defaults ported from the reference implementation's critic
network + get_config() defaults (agents/acfql.py): hidden_dims=(512,512,512,512),
layer_norm=True, num_qs=2.
"""

import flax.nnx as nnx
import jax.numpy as jnp


class _QMLP(nnx.Module):
    """A single Q-value head: MLP with optional LayerNorm + ReLU, ending in a
    scalar output. dict-of-submodules keyed by str(i), matching the
    convention already used for stacked blocks in ac_predictor_nnx.py."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], *, layer_norm: bool, rngs: nnx.Rngs):
        dims = [input_dim, *hidden_dims]
        self.layers = {str(i): nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)}
        self.norms = {str(i): nnx.LayerNorm(d, rngs=rngs) for i, d in enumerate(hidden_dims)} if layer_norm else {}
        self.out = nnx.Linear(hidden_dims[-1], 1, rngs=rngs)
        self._num_layers = len(hidden_dims)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self._num_layers):
            x = self.layers[str(i)](x)
            if str(i) in self.norms:
                x = self.norms[str(i)](x)
            x = nnx.relu(x)
        return self.out(x)[..., 0]  # [B]


class QChunkCritic(nnx.Module):
    """Ensemble of `num_qs` independent Q-MLPs sharing the same input.
    __call__ returns [num_qs, B] (matching the reference's critic output
    convention, e.g. `next_qs.min(axis=0)` in acfql.py's critic_loss)."""

    def __init__(
        self,
        embed_dim: int,
        proprio_dim: int,
        action_dim: int,
        horizon_length: int,
        *,
        hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
        num_qs: int = 2,
        layer_norm: bool = True,
        rngs: nnx.Rngs,
    ):
        self.num_qs = num_qs
        input_dim = embed_dim + proprio_dim + action_dim * horizon_length
        self.heads = {
            str(i): _QMLP(input_dim, hidden_dims, layer_norm=layer_norm, rngs=rngs) for i in range(num_qs)
        }

    def __call__(self, embed: jnp.ndarray, proprio: jnp.ndarray, action_chunk: jnp.ndarray) -> jnp.ndarray:
        """embed: [B, embed_dim], proprio: [B, proprio_dim],
        action_chunk: [B, horizon_length, action_dim]. Returns [num_qs, B]."""
        flat_actions = action_chunk.reshape(action_chunk.shape[0], -1)
        x = jnp.concatenate([embed, proprio, flat_actions], axis=-1)
        return jnp.stack([self.heads[str(i)](x) for i in range(self.num_qs)], axis=0)
