import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from ac_predictor_nnx import VisionTransformerPredictorAC
from convert_checkpoint import convert, load_pytorch_predictor_state, flatten_nnx_state

def verify_conversion(ckpt_path: str):
    print("Initializing NNX Model Architecture...")
    rngs = nnx.Rngs(0)
    # Using your ViT-g predictor defaults
    model = VisionTransformerPredictorAC(
        img_size=(256, 256),
        patch_size=16,
        num_frames=8,
        tubelet_size=2,
        embed_dim=1408,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        rngs=rngs,
    )

    print(f"Loading raw PyTorch state dict from {ckpt_path}...")
    torch_state = load_pytorch_predictor_state(ckpt_path)

    print("Converting and injecting weights into NNX graph...")
    # This runs your conversion function to populate the model in-place
    model = convert(model, torch_state, dry_run=False)
    model.train(False)  # CRITICAL: Freeze stochastic layers/dropout

    # ---------------------------------------------------------
    # VERIFICATION 1: Weight Summary Sanity Check
    # ---------------------------------------------------------
    print("\n--- Running Verification 1: Weight Statistics ---")
    nnx_flat = flatten_nnx_state(model)
    
    # Pick a sample square weight matrix to verify value mapping
    sample_nnx_key = "predictor_blocks.0.attn.qkv.kernel"
    
    if sample_nnx_key in nnx_flat:
        nnx_w = np.array(nnx_flat[sample_nnx_key])
        # Find corresponding source key in torch_state
        # (Based on your regex, it maps from predictor_blocks.0.attn.qkv.weight)
        torch_w = torch_state["predictor_blocks.0.attn.qkv.weight"]
        
        print(f"Checking {sample_nnx_key}:")
        print(f"  PyTorch source shape: {torch_w.shape} -> JAX target shape: {nnx_w.shape}")
        print(f"  PyTorch raw mean:    {torch_w.mean():.6f} | JAX loaded mean:    {nnx_w.mean():.6f}")
        print(f"  PyTorch raw std:     {torch_w.std():.6f} | JAX loaded std:     {nnx_w.std():.6f}")
        
        # Check if the values actually match up to transposition
        if not np.allclose(nnx_w, torch_w.T, atol=1e-6):
            print("❌ ERROR: Values do not align with expected transposition!")
        else:
            print("✅ Weight values match raw checkpoint values perfectly.")

    # ---------------------------------------------------------
    # VERIFICATION 2: Activation Propagation Profile
    # ---------------------------------------------------------
    print("\n--- Running Verification 2: Activation Profiling ---")
    
    # Generate static dummy inputs (matching ViT-g token shapes)
    # batch=2, tokens=64 (example context size), embed_dim=1408
    np.random.seed(42)
    dummy_z_context = jnp.array(np.random.normal(size=(2, 64, 1408)).astype(np.float32))
    dummy_actions = jnp.array(np.random.normal(size=(2, 14)).astype(np.float32))
    dummy_states = jnp.array(np.random.normal(size=(2, 14)).astype(np.float32))

    try:
        # Run a forward pass
        out = model(dummy_z_context, dummy_actions, dummy_states)
        
        # Extract activation stats
        out_np = np.array(out)
        has_nan = np.isnan(out_np).any()
        has_inf = np.isinf(out_np).any()
        out_mean = out_np.mean()
        out_std = out_np.std()

        print("Forward Pass Metrics:")
        print(f"  Output Shape:     {out_np.shape}")
        print(f"  Contains NaNs:    {has_nan}")
        print(f"  Contains Infs:    {has_inf}")
        print(f"  Activation Mean:  {out_mean:.4f}")
        print(f"  Activation Std:   {out_std:.4f}")

        if has_nan or has_inf:
            print("❌ FAILURE: Model produced NaNs/Infs. A square matrix transposition is likely broken.")
        elif out_std > 100.0 or out_std < 1e-4:
            print("❌ FAILURE: Abnormal variance scaling. Check LayerNorm epsilon or residual scales.")
        else:
            print("🚀 SUCCESS: Forward pass is stable! Structural conversion looks solid.")

    except Exception as e:
        print(f"❌ CRITICAL EXPANSION ERROR: Forward pass crashed with: {e}")

if __name__ == "__main__":
    # Point this to your actual file
    verify_conversion("vjepa2-ac-vitg.pt")