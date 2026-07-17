"""Convert a pretrained V-JEPA2-AC predictor PyTorch checkpoint into the NNX
port in ac_predictor_nnx.py.

WORKFLOW (do this in order -- do not skip the dry run):

    python convert_checkpoint.py --ckpt /path/to/vjepa2-ac-vitg.pt --dry-run

This prints:
  (a) every key/shape in the PyTorch predictor state_dict
  (b) every param path/shape in the freshly-initialized NNX model
  (c) which PyTorch keys the regex mapping below could NOT match to an NNX
      path, and vice versa

If (c) is non-empty, the block internals (attn/mlp naming) in modules_nnx.py
don't match the real modules.py 1:1, and KEY_MAP below needs adjusting before
you trust any loaded weights. This is the verification step that replaces
needing the original source file directly.

Once the dry run reports zero unmapped keys on both sides, re-run without
--dry-run to actually write the converted NNX state to disk.
"""

import argparse
import re

import jax.numpy as jnp
import numpy as np
from flax import nnx

from ac_predictor_nnx import VisionTransformerPredictorAC


# --------------------------------------------------------------------------- #
# Regex-based key mapping: (pytorch_key_regex, nnx_path_template, transpose)
# `transpose=True` for nn.Linear.weight, since torch stores [out, in] and
# flax/nnx Linear kernels are stored [in, out].
# --------------------------------------------------------------------------- #
KEY_MAP = [
    (r"^predictor_embed\.weight$", "predictor_embed.kernel", True),
    (r"^predictor_embed\.bias$", "predictor_embed.bias", False),
    (r"^action_encoder\.weight$", "action_encoder.kernel", True),
    (r"^action_encoder\.bias$", "action_encoder.bias", False),
    (r"^state_encoder\.weight$", "state_encoder.kernel", True),
    (r"^state_encoder\.bias$", "state_encoder.bias", False),
    (r"^extrinsics_encoder\.weight$", "extrinsics_encoder.kernel", True),
    (r"^extrinsics_encoder\.bias$", "extrinsics_encoder.bias", False),
    (r"^predictor_norm\.weight$", "predictor_norm.scale", False),
    (r"^predictor_norm\.bias$", "predictor_norm.bias", False),
    (r"^predictor_proj\.weight$", "predictor_proj.kernel", True),
    (r"^predictor_proj\.bias$", "predictor_proj.bias", False),
    # Per-block. ASSUMES standard timm-style naming: norm1/attn.qkv/attn.proj/norm2/mlp.fc1/mlp.fc2
    (r"^predictor_blocks\.(\d+)\.norm1\.weight$", r"predictor_blocks.\1.norm1.scale", False),
    (r"^predictor_blocks\.(\d+)\.norm1\.bias$", r"predictor_blocks.\1.norm1.bias", False),
    (r"^predictor_blocks\.(\d+)\.attn\.qkv\.weight$", r"predictor_blocks.\1.attn.qkv.kernel", True),
    (r"^predictor_blocks\.(\d+)\.attn\.qkv\.bias$", r"predictor_blocks.\1.attn.qkv.bias", False),
    (r"^predictor_blocks\.(\d+)\.attn\.proj\.weight$", r"predictor_blocks.\1.attn.proj.kernel", True),
    (r"^predictor_blocks\.(\d+)\.attn\.proj\.bias$", r"predictor_blocks.\1.attn.proj.bias", False),
    (r"^predictor_blocks\.(\d+)\.norm2\.weight$", r"predictor_blocks.\1.norm2.scale", False),
    (r"^predictor_blocks\.(\d+)\.norm2\.bias$", r"predictor_blocks.\1.norm2.bias", False),
    (r"^predictor_blocks\.(\d+)\.mlp\.fc1\.weight$", r"predictor_blocks.\1.mlp.fc1.kernel", True),
    (r"^predictor_blocks\.(\d+)\.mlp\.fc1\.bias$", r"predictor_blocks.\1.mlp.fc1.bias", False),
    (r"^predictor_blocks\.(\d+)\.mlp\.fc2\.weight$", r"predictor_blocks.\1.mlp.fc2.kernel", True),
    (r"^predictor_blocks\.(\d+)\.mlp\.fc2\.bias$", r"predictor_blocks.\1.mlp.fc2.bias", False),
]


def map_pytorch_key(key: str):
    for pattern, template, transpose in KEY_MAP:
        m = re.match(pattern, key)
        if m:
            return re.sub(pattern, template, key), transpose
    return None, None


def flatten_nnx_state(model) -> dict:
    """Return {'a.b.c': array} for every leaf param in the NNX model."""
    state = nnx.state(model, nnx.Param)
    flat = {}

    def _walk(prefix, node):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif hasattr(node, "value"):
            flat[prefix] = node.value
        else:
            # nnx.State leaves are usually VariableState-like with .value;
            # if this branch triggers, print(type(node)) to adjust.
            flat[prefix] = node

    _walk("", state.to_pure_dict() if hasattr(state, "to_pure_dict") else state)
    return flat


def load_pytorch_predictor_state(ckpt_path: str) -> dict:
    import torch  # local import: only needed on the machine doing conversion

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Checkpoints from this repo commonly nest the predictor under a top-level
    # key (e.g. "predictor") and prefix module names with "module." (DDP).
    # Adjust here based on what --dry-run shows you for your actual file.
    if "predictor" in raw:
        sd = raw["predictor"]
    elif "target_predictor" in raw:
        sd = raw["target_predictor"]
    else:
        sd = raw
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    return {k: v.numpy() for k, v in sd.items()}


def convert(model, torch_state: dict, dry_run: bool):
    nnx_flat = flatten_nnx_state(model)
    nnx_paths = set(nnx_flat.keys())

    mapped = {}
    unmapped_src = []
    for k, v in torch_state.items():
        target, transpose = map_pytorch_key(k)
        if target is None:
            unmapped_src.append(k)
            continue
        arr = np.transpose(v) if transpose else v
        mapped[target] = (arr, k)

    unmapped_dst = sorted(nnx_paths - set(mapped.keys()))
    shape_mismatches = []
    for target, (arr, src_key) in mapped.items():
        if target in nnx_flat and tuple(arr.shape) != tuple(nnx_flat[target].shape):
            shape_mismatches.append((src_key, target, arr.shape, nnx_flat[target].shape))

    print(f"PyTorch keys total: {len(torch_state)}")
    print(f"NNX param paths total: {len(nnx_paths)}")
    print(f"Mapped: {len(mapped)}")
    print(f"Unmapped PyTorch keys ({len(unmapped_src)}):")
    for k in unmapped_src[:40]:
        print(f"  {k}  shape={tuple(torch_state[k].shape)}")
    print(f"NNX paths with no source ({len(unmapped_dst)}):")
    for k in unmapped_dst[:40]:
        print(f"  {k}  shape={tuple(nnx_flat[k].shape)}")
    print(f"Shape mismatches ({len(shape_mismatches)}):")
    for src, dst, s1, s2 in shape_mismatches[:40]:
        print(f"  {src} -> {dst}: pytorch {s1} vs nnx {s2}")

    if dry_run:
        print("\nDry run only -- no weights written. Fix KEY_MAP / modules_nnx.py "
              "until unmapped/mismatch lists above are empty, then re-run without --dry-run.")
        return None

    if unmapped_src or unmapped_dst or shape_mismatches:
        raise RuntimeError(
            "Refusing to load: mapping is incomplete or has shape mismatches. "
            "Run with --dry-run and fix KEY_MAP / modules_nnx.py first."
        )

    # Write values into the model in place.
    graphdef, state = nnx.split(model)
    pure = state.to_pure_dict()

    def _set(d, path, value):
        parts = path.split(".")
        for p in parts[:-1]:
            d = d[p] if p in d else d[int(p)]
        leaf_key = parts[-1]
        target = d[leaf_key] if leaf_key in d else d[int(leaf_key)]
        target.value = jnp.asarray(value)

    for target_path, (arr, _src) in mapped.items():
        _set(pure, target_path, arr)

    new_model = nnx.merge(graphdef, pure)
    print("Loaded pretrained predictor weights into NNX model.")
    return new_model


def save_converted_state(model, path: str):
    """Save a converted (pretrained-weights-loaded) predictor's params to disk
    as a flat .npz, for later use by load_and_merge_predictor_state(). This
    was missing before -- convert() built the loaded model in memory but
    nothing ever persisted it, so init_train_state had no way to pick it up."""
    flat = flatten_nnx_state(model)
    # '.' -> '__' because npz member names are written as f"{key}.npy" internally;
    # dots in the key make that ambiguous with the extension.
    np.savez(path, **{k.replace(".", "__"): np.asarray(v) for k, v in flat.items()})
    print(f"Saved converted predictor state to {path} ({len(flat)} arrays)")


def load_and_merge_predictor_state(full_model, npz_path: str):
    """Load a .npz produced by save_converted_state() and merge it into
    `full_model.jepa_predictor` in place (full_model is expected to be an
    OpenPIWithJEPA instance, not a standalone predictor -- paths get prefixed
    with 'jepa_predictor.' accordingly). Returns the merged model.
    Call this from init_train_state() after constructing OpenPIWithJEPA and
    before extracting final params, if config.jepa_predictor_checkpoint is set.
    """
    npz = np.load(npz_path)
    flat = {k.replace("__", "."): v for k, v in npz.items()}

    graphdef, state = nnx.split(full_model)
    pure = state.to_pure_dict()

    def _set(d, path_str, value):
        parts = path_str.split(".")
        for p in parts[:-1]:
            d = d[p] if p in d else d[int(p)]
        leaf_key = parts[-1]
        target = d[leaf_key] if leaf_key in d else d[int(leaf_key)]
        target.value = jnp.asarray(value)

    missing = []
    for k, v in flat.items():
        try:
            _set(pure, f"jepa_predictor.{k}", v)
        except (KeyError, IndexError):
            missing.append(k)
    if missing:
        raise RuntimeError(
            f"{len(missing)} keys from {npz_path} did not map onto full_model.jepa_predictor "
            f"(e.g. {missing[:5]}). This usually means the predictor config used when building "
            f"full_model doesn't match the one used during conversion."
        )
    print(f"Merged {len(flat)} converted predictor params into full_model.jepa_predictor.")
    return nnx.merge(graphdef, pure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to vjepa2-ac-*.pt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-path", type=str, default=None,
                         help="Where to write the converted predictor state (.npz). "
                              "Required unless --dry-run. Load it back via "
                              "load_and_merge_predictor_state() in init_train_state.")
    # Model config must match the checkpoint's architecture (defaults below
    # match the ViT-g/16 predictor per the paper: depth=24, num_heads=16,
    # predictor_embed_dim=1024). Adjust for other checkpoint variants.
    parser.add_argument("--embed-dim", type=int, default=1408)  # ViT-g encoder dim
    parser.add_argument("--predictor-embed-dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    args = parser.parse_args()

    rngs = nnx.Rngs(0)
    model = VisionTransformerPredictorAC(
        img_size=(args.img_size, args.img_size),
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        embed_dim=args.embed_dim,
        predictor_embed_dim=args.predictor_embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        rngs=rngs,
    )

    torch_state = load_pytorch_predictor_state(args.ckpt)
    converted = convert(model, torch_state, dry_run=args.dry_run)
    if not args.dry_run:
        if not args.save_path:
            raise ValueError("--save-path is required unless --dry-run is set.")
        save_converted_state(converted, args.save_path)