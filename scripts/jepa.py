import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from jepa_modules import ACBlock, build_action_block_causal_attention_mask

# ==========================================
# 1. Integration of the VisionTransformerPredictorAC Wrapper
# ==========================================
class VisionTransformerPredictorAC(nn.Module):
    """Action Conditioned Vision Transformer Predictor matching Meta V-JEPA 2"""

    def __init__(
        self,
        img_size=(256, 256),
        patch_size=16,
        num_frames=4,
        embed_dim=1152,           # Matches PaliGemma/VLA output projection
        predictor_embed_dim=1024, # Inner embedding dimension
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        action_embed_dim=8,       # 8D Relative Action Chunk
        num_add_tokens=1,         # 1 action token per frame bucket
        use_rope=True,
        **kwargs
    ):
        super().__init__()
        self.num_frames = num_frames
        self.is_frame_causal = True

        # Map input to predictor dimensions
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size

        self.grid_height = self.img_height // self.patch_size
        self.grid_width = self.img_width // self.patch_size
        self.num_add_tokens = num_add_tokens

        # Attention Blocks matching provided source
        self.predictor_blocks = nn.ModuleList([
            ACBlock(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                act_layer=nn.GELU,
                norm_layer=norm_layer,
                use_rope=use_rope,
                grid_size=self.grid_height,
                is_causal=False, # Masking is explicitly controlled via attn_mask
            )
            for _ in range(depth)
        ])

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # Build causal attention mask using your true function signature
        self.register_buffer("attn_mask", build_action_block_causal_attention_mask(
            T=self.num_frames, 
            H=self.grid_height, 
            W=self.grid_width, 
            add_tokens=num_add_tokens
        ))

    def forward(self, x, actions):
        B, N_patches, D_in = x.size()
        T = self.num_frames

        # Project sizes
        x = self.predictor_embed(x)
        D = x.size(-1)

        x = x.unsqueeze(1).repeat(1, T, 1, 1)
        a = self.action_encoder(actions).view(B, T, self.num_add_tokens, D)
        x = torch.cat([a, x], dim=2).flatten(1, 2)

        # FIX: Lazy-cache the mask on the correct device instead of rebuilding it
        if not hasattr(self, "_cached_attn_mask") or self._cached_attn_mask.size(0) != x.size(1):
            cam_multiplier = N_patches // (self.grid_height * self.grid_width)
            effective_width = self.grid_width * cam_multiplier
            
            # Build once
            raw_mask = build_action_block_causal_attention_mask(
                T=T, H=self.grid_height, W=effective_width, add_tokens=self.num_add_tokens
            )
            self._cached_attn_mask = raw_mask[:x.size(1), :x.size(1)].to(x.device)
            self._effective_width = effective_width

        # Retrieve cached elements instantly
        attn_mask = self._cached_attn_mask
        effective_width = self._effective_width
        # -----------------------------------------------------------------

        # 5. Process sequence layers through the Meta blocks
        for blk in self.predictor_blocks:
            x = blk(
                x,
                mask=None,
                attn_mask=attn_mask,
                T=T,
                H=self.grid_height,
                W=effective_width,
                action_tokens=self.num_add_tokens,
            )

        # 6. Extract vision sequence out, stripping action slots
        # FIX: Also update the unflattening shape right below the block loop
        x = x.view(B, T, self.num_add_tokens + N_patches, D)
        x = x[:, :, self.num_add_tokens :, :].flatten(1, 2)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        return x


# ==========================================
# 2. Core VLA Exploration Workspace Pipeline
# ==========================================
class VLAExplorationSystem(nn.Module):
    def __init__(self, vision_tokens=256, horizon=4):
        super().__init__()
        self.vision_tokens = vision_tokens
        
        self.predictor = VisionTransformerPredictorAC(
            img_size=(256, 256),
            patch_size=16,
            num_frames=horizon,
            embed_dim=1152,
            predictor_embed_dim=1024,
            depth=6,
            num_heads=16,
            action_embed_dim=8,  # Relative 8D changes
            num_add_tokens=1
        )

    def get_intrinsic_reward(self, prefix_tokens_t, actions_t, prefix_tokens_next):
        """
        Computes L2 predictive curiosity over latent states.
        Removes language tokens to focus solely on workspace state transitions.
        """
        # Strip out prompt tokens, isolating target workspace visuals
        z_t = prefix_tokens_t[:, :self.vision_tokens, :]
        y_next = prefix_tokens_next[:, :self.vision_tokens*4, :]

        # Predict future frame features conditioned on action
        z_next_pred = self.predictor(z_t, actions_t)

        # Compute MSE across hidden dimension
        squared_errors = (z_next_pred - y_next) ** 2
        
        # Mean-pool across the sequence spatial grid
        intrinsic_reward = torch.mean(squared_errors, dim=[-2, -1]) # [B]
        return intrinsic_reward


# ==========================================
# 3. Execution Verification Test
# ==========================================
if __name__ == "__main__":
    # Settings modeling a batch run from your current training pipeline
    B = 2
    T = 4
    num_patches = 256
    vla_dim = 1152

    # Instantiate exploration block
    system = VLAExplorationSystem(vision_tokens=num_patches, horizon=T).cuda()
    system.eval()

    # Create dummy multi-modal tokens (e.g. 256 visual + 32 description tokens)
    simulated_vla_out_t = torch.randn(B, num_patches + 32, vla_dim).cuda()
    simulated_vla_out_next = torch.randn(B, (T * num_patches) + 32, vla_dim).cuda()    
    # 8D Relative Action chunks across target horizon window
    simulated_actions = torch.randn(B, T, 8).cuda()

    with torch.no_grad():
        rewards = system.get_intrinsic_reward(
            simulated_vla_out_t, 
            simulated_actions, 
            simulated_vla_out_next
        )

    print("\n--- Shape Signatures Verified Natively ---")
    print(f"Action input shape:       {simulated_actions.shape}")
    print(f"Computed reward tensor:   {rewards.shape}")
    print(f"Batch values:             {rewards.cpu().numpy()}")