# Flax NNX port of facebookresearch/vjepa2 src/models/utils/modules.py (ACBlock and friends).
#
# IMPORTANT PROVENANCE NOTE:
# The original `modules.py` source could not be fetched directly (GitHub blocks
# automated access to that code-browser path). This port is reconstructed from:
#   - the VisionTransformerPredictorAC source you provided (confirms `attn.proj`
#     and `mlp.fc2` attribute names via `_rescale_blocks`)
#   - the V-JEPA2 paper (arxiv.org/abs/2506.09985): 24 layers, 16 heads, 1024
#     hidden dim, GELU, 3D-RoPE for patch tokens / 1D (temporal-only) RoPE for
#     action-state tokens, block-causal attention (full attention within a
#     frame, causal across frames)
#   - Meta's standard ViT block convention used across their JEPA repos
#     (norm1 -> attn.qkv/attn.proj -> norm2 -> mlp.fc1/mlp.fc2)
#
# WHAT IS NOT VERIFIED: the exact RoPE frequency convention (theta base, how
# head_dim is split across the t/h/w axes) and the precise implementation of
# `build_action_block_causal_attention_mask`. These have no learnable
# parameters, so a pretrained checkpoint will *load* successfully even if
# these details are off -- but the numerics of the loaded model may not match
# Meta's original outputs. Before trusting this for fine-tuning, run the
# parity check described in convert_checkpoint.py (compare this module's
# output against the original PyTorch predictor on identical inputs, if you
# have PyTorch + the original repo available on some machine to cross-check).

import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx


def trunc_normal_init(std: float = 0.02):
    """Equivalent to torch's trunc_normal_(std=std) used in _init_weights."""
    return jax.nn.initializers.truncated_normal(stddev=std)


# --------------------------------------------------------------------------- #
# 3D / 1D Rotary Position Embeddings
# --------------------------------------------------------------------------- #

def _rope_freqs(dim: int, theta: float = 10000.0) -> jnp.ndarray:
    """Standard RoPE inverse-frequency bank for a sub-dimension `dim` (must be even)."""
    assert dim % 2 == 0, f"RoPE sub-dimension must be even, got {dim}"
    exponents = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    return 1.0 / (theta ** exponents)  # [dim/2]


def _apply_rope_1axis(x: jnp.ndarray, pos: jnp.ndarray, freqs: jnp.ndarray) -> jnp.ndarray:
    """Rotate the last-dim pairs of `x` by angle pos*freqs.
    x: [..., N, dim] (dim even, restricted to one axis's sub-dimension)
    pos: [..., N] integer/float positions for that axis
    freqs: [dim/2] inverse frequency bank
    """
    angles = pos[..., None] * freqs  # [..., N, dim/2]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    x1, x2 = jnp.split(x, 2, axis=-1)  # each [..., N, dim/2]
    rotated = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated


def build_3d_rope_coords(T: int, H: int, W: int, action_tokens: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-token (t, h, w) coordinates for one frame-block-interleaved sequence
    of length T * (action_tokens + H*W).

    Conditioning tokens (the first `action_tokens` slots of every frame) get
    h = w = 0, which makes their h/w rotary components an identity rotation --
    i.e. only their temporal component is meaningful, matching the paper's
    "temporal-only RoPE for action/pose tokens" description.
    """
    frame_len = action_tokens + H * W
    t_patch, h_patch, w_patch = jnp.meshgrid(
        jnp.arange(H), jnp.arange(W), indexing="ij"
    )
    h_patch = h_patch.reshape(-1)  # [H*W]
    w_patch = w_patch.reshape(-1)  # [H*W]

    t_idx = []
    h_idx = []
    w_idx = []
    for t in range(T):
        # conditioning tokens
        t_idx.append(jnp.full((action_tokens,), t, dtype=jnp.float32))
        h_idx.append(jnp.zeros((action_tokens,), dtype=jnp.float32))
        w_idx.append(jnp.zeros((action_tokens,), dtype=jnp.float32))
        # patch tokens
        t_idx.append(jnp.full((H * W,), t, dtype=jnp.float32))
        h_idx.append(h_patch.astype(jnp.float32))
        w_idx.append(w_patch.astype(jnp.float32))
    t_idx = jnp.concatenate(t_idx)  # [T * frame_len]
    h_idx = jnp.concatenate(h_idx)
    w_idx = jnp.concatenate(w_idx)
    assert t_idx.shape[0] == T * frame_len
    return t_idx, h_idx, w_idx


def apply_3d_rope(q_or_k: jnp.ndarray, t_idx: jnp.ndarray, h_idx: jnp.ndarray,
                   w_idx: jnp.ndarray, theta: float = 10000.0) -> jnp.ndarray:
    """q_or_k: [B, num_heads, N, head_dim]. Splits head_dim into three equal
    chunks for (t, h, w) axes -- ASSUMED equal split; adjust if your checkpoint
    implies otherwise (e.g. via parity-testing against the original repo)."""
    head_dim = q_or_k.shape[-1]
    third = head_dim // 3
    # round down to even for valid rope pairing; leftover dims pass through unrotated
    third = third - (third % 2)
    d_t, d_h, d_w = third, third, head_dim - 2 * third
    d_w = d_w - (d_w % 2)
    remainder = head_dim - d_t - d_h - d_w

    xt, xh, xw, xr = jnp.split(q_or_k, [d_t, d_t + d_h, d_t + d_h + d_w], axis=-1)

    freqs_t = _rope_freqs(d_t, theta)
    freqs_h = _rope_freqs(d_h, theta)
    freqs_w = _rope_freqs(d_w, theta)

    xt = _apply_rope_1axis(xt, t_idx[None, None, :], freqs_t)
    xh = _apply_rope_1axis(xh, h_idx[None, None, :], freqs_h)
    xw = _apply_rope_1axis(xw, w_idx[None, None, :], freqs_w)

    if remainder > 0:
        return jnp.concatenate([xt, xh, xw, xr], axis=-1)
    return jnp.concatenate([xt, xh, xw], axis=-1)


# --------------------------------------------------------------------------- #
# Block-causal action mask
# --------------------------------------------------------------------------- #

def build_action_block_causal_attention_mask(
    grid_depth: int, grid_height: int, grid_width: int, add_tokens: int = 2
) -> jnp.ndarray:
    """Boolean mask [S, S], True where attention is ALLOWED.
    Full attention within a frame block; causal (current + past frames only)
    across frame blocks. S = grid_depth * (grid_height*grid_width + add_tokens).
    """
    frame_len = grid_height * grid_width + add_tokens
    S = grid_depth * frame_len
    frame_of = jnp.repeat(jnp.arange(grid_depth), frame_len)  # [S]
    # query i may attend to key j iff frame_of(j) <= frame_of(i)
    allowed = frame_of[:, None] >= frame_of[None, :]  # [S, S]
    return allowed


# --------------------------------------------------------------------------- #
# Attention + MLP + ACBlock
# --------------------------------------------------------------------------- #

class ACAttention(nnx.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, qk_scale: Optional[float],
                 attn_drop: float, proj_drop: float, use_rope: bool, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.use_rope = use_rope

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias,
                               kernel_init=trunc_normal_init(), rngs=rngs)
        self.proj = nnx.Linear(dim, dim, use_bias=True,
                                kernel_init=trunc_normal_init(), rngs=rngs)
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

    def __call__(self, x, attn_mask=None, T=None, H=None, W=None, action_tokens=0,
                 deterministic: bool = True):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope and T is not None:
            t_idx, h_idx, w_idx = build_3d_rope_coords(T, H, W, action_tokens)
            q = apply_3d_rope(q, t_idx, h_idx, w_idx)
            k = apply_3d_rope(k, t_idx, h_idx, w_idx)

        attn_logits = jnp.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        if attn_mask is not None:
            bias = jnp.where(attn_mask, 0.0, jnp.finfo(attn_logits.dtype).min)
            attn_logits = attn_logits + bias[None, None, :, :]
        attn = jax.nn.softmax(attn_logits, axis=-1)
        attn = self.attn_drop(attn, deterministic=deterministic)

        out = jnp.einsum("bhnm,bhmd->bhnd", attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out, deterministic=deterministic)
        return out


class Mlp(nnx.Module):
    def __init__(self, in_dim: int, hidden_dim: int, act_layer, drop: float, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden_dim, kernel_init=trunc_normal_init(), rngs=rngs)
        self.act = act_layer
        self.fc2 = nnx.Linear(hidden_dim, in_dim, kernel_init=trunc_normal_init(), rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x, deterministic: bool = True):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, deterministic=deterministic)
        x = self.fc2(x)
        x = self.drop(x, deterministic=deterministic)
        return x


class ACBlock(nnx.Module):
    """Action-conditioned transformer block. Mirrors the PyTorch ACBlock's
    public call signature: blk(x, mask=None, attn_mask=..., T=..., H=..., W=...,
    action_tokens=...)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool,
                 qk_scale: Optional[float], drop: float, attn_drop: float, drop_path: float,
                 act_layer, norm_layer, use_rope: bool, grid_size: int, wide_silu: bool,
                 rngs: nnx.Rngs):
        self.norm1 = norm_layer(dim, rngs=rngs)
        self.attn = ACAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop,
                                 use_rope, rngs=rngs)
        self.norm2 = norm_layer(dim, rngs=rngs)
        mlp_hidden = int(dim * mlp_ratio)
        if wide_silu and act_layer is nnx.silu:
            mlp_hidden = int(mlp_hidden * 2 / 3)  # SwiGLU-style width correction, common convention
        self.mlp = Mlp(dim, mlp_hidden, act_layer, drop, rngs=rngs)
        # NOTE: drop_path (stochastic depth) omitted -- negligible at fine-tuning
        # time with small LR; add nnx-side stochastic depth if you need exact parity.

    def __call__(self, x, mask=None, attn_mask=None, T=None, H=None, W=None,
                 action_tokens=0, deterministic: bool = True):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, T=T, H=H, W=W,
                           action_tokens=action_tokens, deterministic=deterministic)
        x = x + self.mlp(self.norm2(x), deterministic=deterministic)
        return x
