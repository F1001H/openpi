"""Flax NNX port of facebookresearch/vjepa2 src/models/ac_predictor.py.

See modules_nnx.py for provenance notes on what is confirmed vs. reconstructed.
Parameter names below (predictor_embed, action_encoder, state_encoder,
extrinsics_encoder, predictor_blocks, predictor_norm, predictor_proj) are taken
verbatim from the PyTorch source you provided, so the top-level structure of
convert_checkpoint.py's key-mapping should be reliable; it's the internals of
each ACBlock (RoPE convention, attention layout) that carry the most risk and
are worth checking against your actual checkpoint (see convert_checkpoint.py).
"""

import functools
import math

import jax.numpy as jnp
from flax import nnx

from jepa.modules_nnx import (
    ACBlock,
    build_action_block_causal_attention_mask,
    trunc_normal_init,
)


class VisionTransformerPredictorAC(nnx.Module):
    def __init__(
        self,
        *,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_epsilon=1e-6,
        init_std=0.02,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_rope=True,
        action_embed_dim=7,
        use_extrinsics=False,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        self.is_frame_causal = is_frame_causal
        self.use_extrinsics = use_extrinsics
        self.init_std = init_std

        kinit = trunc_normal_init(init_std)

        self.predictor_embed = nnx.Linear(embed_dim, predictor_embed_dim, use_bias=True,
                                           kernel_init=kinit, rngs=rngs)
        self.action_encoder = nnx.Linear(action_embed_dim, predictor_embed_dim, use_bias=True,
                                          kernel_init=kinit, rngs=rngs)
        self.state_encoder = nnx.Linear(action_embed_dim, predictor_embed_dim, use_bias=True,
                                         kernel_init=kinit, rngs=rngs)
        self.extrinsics_encoder = nnx.Linear(action_embed_dim - 1, predictor_embed_dim, use_bias=True,
                                              kernel_init=kinit, rngs=rngs)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size

        act_layer = nnx.silu if use_silu else nnx.gelu
        norm_layer = functools.partial(nnx.LayerNorm, epsilon=norm_epsilon)

        self.predictor_blocks = {
            str(i): ACBlock(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.0,  # see note in modules_nnx.ACBlock
                act_layer=act_layer,
                norm_layer=norm_layer,
                use_rope=use_rope,
                grid_size=self.grid_height,
                wide_silu=wide_silu,
                rngs=rngs,
            )
            for i in range(depth)
        }
        self._depth = depth

        self.predictor_norm = norm_layer(predictor_embed_dim, rngs=rngs)
        self.predictor_proj = nnx.Linear(predictor_embed_dim, embed_dim, use_bias=True,
                                          kernel_init=kinit, rngs=rngs)

        self._rescale_blocks()

        self.attn_mask = None
        if self.is_frame_causal:
            grid_depth = self.num_frames // self.tubelet_size
            add_tokens = 3 if use_extrinsics else 2
            mask = build_action_block_causal_attention_mask(
                grid_depth, self.grid_height, self.grid_width, add_tokens
            )
            # Wrap as a non-Param nnx.Variable: raw jax/numpy arrays can't be
            # bare Module attributes in NNX (nnx.state/split/merge will raise
            # "Array leaves are not supported"). Using the generic Variable
            # (not nnx.Param) keeps it out of trainable_filter/freeze_filter,
            # which target nnx.Param specifically.
            self.attn_mask = nnx.Variable(mask)

    def _rescale_blocks(self):
        """Mirrors the PyTorch _rescale_blocks: divide attn.proj and mlp.fc2
        kernels by sqrt(2 * layer_id) post-init."""
        for i in range(self._depth):
            blk = self.predictor_blocks[str(i)]
            scale = math.sqrt(2.0 * (i + 1))
            blk.attn.proj.kernel.value = blk.attn.proj.kernel.value / scale
            blk.mlp.fc2.kernel.value = blk.mlp.fc2.kernel.value / scale

    def __call__(self, x, actions, states, extrinsics=None, deterministic: bool = True):
        """
        x: [B, N_ctxt, embed_dim] context tokens (already flattened over T*H*W)
        actions: [B, T, action_embed_dim]
        states:  [B, T, action_embed_dim]
        extrinsics: [B, T, action_embed_dim - 1] if use_extrinsics
        """
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.shape
        T = N_ctxt // (self.grid_height * self.grid_width)

        s = jnp.expand_dims(self.state_encoder(states), axis=2)   # [B, T, 1, D]
        a = jnp.expand_dims(self.action_encoder(actions), axis=2)  # [B, T, 1, D]
        x = x.reshape(B, T, self.grid_height * self.grid_width, D)

        if self.use_extrinsics:
            e = jnp.expand_dims(self.extrinsics_encoder(extrinsics), axis=2)
            x = jnp.concatenate([a, s, e, x], axis=2).reshape(B, -1, D)
        else:
            x = jnp.concatenate([a, s, x], axis=2).reshape(B, -1, D)

        cond_tokens = 3 if self.use_extrinsics else 2
        attn_mask = (
            self.attn_mask.value[: x.shape[1], : x.shape[1]]
            if self.attn_mask is not None
            else None
        )

        for i in range(self._depth):
            blk = self.predictor_blocks[str(i)]
            x = blk(x, mask=None, attn_mask=attn_mask, T=T, H=self.grid_height,
                     W=self.grid_width, action_tokens=cond_tokens, deterministic=deterministic)

        x = x.reshape(B, T, cond_tokens + self.grid_height * self.grid_width, D)
        x = x[:, :, cond_tokens:, :].reshape(B, -1, D)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x


def vit_ac_predictor(rngs: nnx.Rngs, **kwargs):
    return VisionTransformerPredictorAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_epsilon=1e-6,
        rngs=rngs,
        **kwargs,
    )