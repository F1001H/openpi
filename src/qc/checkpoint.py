"""Loading a trained QChunkCritic checkpoint (saved by
scripts/train_qc_critic.py). Uses the plain-nnx.State orbax pattern (save/
restore the State directly against an abstract template, not
to_pure_dict()/replace_by_pure_dict) -- verified locally to round-trip
correctly (save -> restore -> nnx.merge -> forward pass), unlike
replace_by_pure_dict which caused real graphdef-mismatch bugs elsewhere in
this repo (see scripts/train_end_to_end.py's init_train_state comments).
"""

import flax.nnx as nnx
import jax
import orbax.checkpoint as ocp

from qc.critic import QChunkCritic


def load_critic(
    checkpoint_path: str,
    embed_dim: int,
    proprio_dim: int,
    action_dim: int,
    horizon_length: int,
    *,
    num_qs: int = 2,
    hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
    layer_norm: bool = True,
    use_target: bool = False,
) -> QChunkCritic:
    """Loads the critic saved at checkpoint_path (a directory ending in e.g.
    .../final, as written by scripts/train_qc_critic.py). Set use_target=True
    to load the (more stable, EMA-smoothed) target network instead of the
    online one -- reasonable for inference-time action scoring."""
    template = QChunkCritic(
        embed_dim, proprio_dim, action_dim, horizon_length,
        hidden_dims=hidden_dims, num_qs=num_qs, layer_norm=layer_norm, rngs=nnx.Rngs(0),
    )
    graphdef, template_state = nnx.split(template)
    abstract_item = {
        "params": jax.eval_shape(lambda: template_state),
        "target_params": jax.eval_shape(lambda: template_state),
        "step": jax.eval_shape(lambda: jax.numpy.array(0)),
    }
    restored = ocp.PyTreeCheckpointer().restore(checkpoint_path, item=abstract_item)
    params = restored["target_params"] if use_target else restored["params"]
    return nnx.merge(graphdef, params)
