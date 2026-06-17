import math
import torch
import torch.nn as nn
from functools import partial

# ==========================================
# 1. Dummy Helper Modules (To replicate your imports)
# ==========================================
class ACBlock(nn.Module):
    """Placeholder for the V-JEPA 2 / starVLA Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, act_layer, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.Linear(dim, dim) # Simplified for structure execution
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    def forward(self, x, attn_mask=None, **kwargs):
        # In production, this computes full or masked MHSA + MLP
        return x + self.mlp(self.norm(x))

def build_action_block_causal_attention_mask(grid_depth, grid_height, grid_width, add_tokens):
    """Builds the 2D block-causal mask to prevent looking into future frames"""
    L_per_frame = add_tokens + (grid_height * grid_width)
    total_L = grid_depth * L_per_frame
    mask = torch.zeros(total_L, total_L)
    for t_i in range(grid_depth):
        # Frame t_i can only attend to frames <= t_i
        mask[t_i * L_per_frame : (t_i + 1) * L_per_frame, : (t_i + 1) * L_per_frame] = 1.0
    return (mask == 0.0) # True means mask out / hide

# ==========================================
# 2. Complete Action-Conditioned JEPA Predictor
# ==========================================
class VisionTransformerPredictorAC(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=1,
        embed_dim=1152,          # Matches PaliGemma/VLA output dim
        predictor_embed_dim=512,  # Lightweight inner bottleneck
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        action_embed_dim=8,       # 8D Relative Action Chunk
        num_add_tokens=1,         # 1 action token per frame bucket
        **kwargs
    ):
        super().__init__()
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.num_add_tokens = num_add_tokens

        # Projection Layers
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim)
        
        # Core Blocks
        self.predictor_blocks = nn.ModuleList([
            ACBlock(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                act_layer=nn.GELU,
            ) for _ in range(depth)
        ])

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

        # Generate Time-Causal Blocking Mask
        grid_depth = self.num_frames // self.tubelet_size
        self.register_buffer("attn_mask", build_action_block_causal_attention_mask(
            grid_depth, self.grid_height, self.grid_width, add_tokens=num_add_tokens
        ))

    def forward(self, x, actions):
        """
        x: Context Visual tokens [B, T * H * W, Embed_Dim] (Stripped of text)
        actions: Relative Action Chunk [B, T, Action_Dim]
        """
        # 1. Project context visual features to inner dimensions
        x = self.predictor_embed(x) 
        B, N_ctxt, D = x.size()
        T = self.num_frames

        # 2. Encode and structure action tokens to match time steps
        a = self.action_encoder(actions) # [B, T, D]
        a = a.unsqueeze(2)               # [B, T, 1, D] (1 action token per frame)

        # 3. Interleave actions and spatial grids per time frame
        x = x.view(B, T, self.grid_height * self.grid_width, D)
        x = torch.cat([a, x], dim=2)     # Interleave: [B, T, 1 + HW, D]
        x = x.flatten(1, 2)              # Flatten to continuous tape: [B, T * (1 + HW), D]

        # Slice causal mask to match current sequence size
        current_L = x.size(1)
        current_mask = self.attn_mask[:current_L, :current_L]

        # 4. Forward through Transformer Layers
        for blk in self.predictor_blocks:
            x = blk(x, Tann_mask=current_mask)
        x = self.predictor_norm(x)

        # 5. Unpack and isolate predicted target visual tokens (discard actions)
        x = x.view(B, T, self.num_add_tokens + self.grid_height * self.grid_width, D)
        x = x[:, :, self.num_add_tokens:, :].flatten(1, 2) # [B, T * H * W, D]
        
        # Project back up to VLA dimension
        return self.predictor_proj(x)

# ==========================================
# 3. Comprehensive Wrapper: VLA-JEPA Pipeline
# ==========================================
class VLA_JEPA_System(nn.Module):
    def __init__(self, num_vision_tokens=256, action_dim=8, horizon=4):
        super().__init__()
        self.num_vision_tokens = num_vision_tokens
        
        # The Action-Conditioned Predictor block
        self.predictor = VisionTransformerPredictorAC(
            num_frames=horizon,
            action_embed_dim=action_dim,
            num_add_tokens=1
        )

    def compute_intrinsic_reward(self, prefix_tokens_t, actions_t, ema_prefix_tokens_next):
        """
        Computes the clean L2 latent residual used for the Q-target step.
        
        prefix_tokens_t: Output of embed_prefix at time t [B, Total_Tokens, 1152]
        actions_t: Relative action chunk executed at time t [B, Horizon, 8]
        ema_prefix_tokens_next: Output of EMA embed_prefix at t+1 [B, Total_Tokens, 1152]
        """
        # 1. Strip out Language Tokens (Keep only visual tokens at front of sequence)
        z_t = prefix_tokens_t[:, :self.num_vision_tokens, :]
        y_next = ema_prefix_tokens_next[:, :self.num_vision_tokens, :]

        # 2. Predict next visual latents conditioned on movement
        z_next_pred = self.predictor(z_t, actions_t)

        # 3. Compute JEPA Prediction Error (L2 loss per sequence item)
        # Higher error = Higher unexpected workspace change = Surge in exploration drive
        prediction_error = torch.mean((z_next_pred - y_next) ** 2, dim=-1) # [B, N_vision]
        
        # Mean-pool across the spatial token patches to yield a clean scalar reward per batch item
        intrinsic_reward = torch.mean(prediction_error, dim=-1) # [B]
        
        return intrinsic_reward

# ==========================================
# 4. Verifying Tensor Shapes (Execution Test)
# ==========================================
if __name__ == "__main__":
    # Setup scenario parameters matching your physical setup
    batch_size = 4
    horizon = 4
    tokens_per_image = 256 # Standard PaliGemma patch count
    text_prompt_tokens = 32
    vla_embed_dim = 1152
    
    # Initialize Pipeline wrapper
    jepa_system = VLA_JEPA_System(num_vision_tokens=tokens_per_image, horizon=horizon)
    jepa_system.eval()

    print("--- Simulating VLA Engine Architecture Slices ---")
    
    # 1. Simulate output of `embed_prefix(obs)` [Image Patches + Prompt Text]
    simulated_prefix_t = torch.randn(batch_size, tokens_per_image + text_prompt_tokens, vla_embed_dim)
    simulated_prefix_next = torch.randn(batch_size, tokens_per_image + text_prompt_tokens, vla_embed_dim)
    
    # 2. Simulate continuous 8D relative action commands generated across horizon
    simulated_actions = torch.randn(batch_size, horizon, 8)
    
    print(f"Input Sequence Shape (with Text Noise): {simulated_prefix_t.shape}")
    print(f"Action Chunk Window Shape:             {simulated_actions.shape}")

    # 3. Process through your pipeline
    with torch.no_grad():
        rewards = jepa_system.compute_intrinsic_reward(
            simulated_prefix_t, 
            simulated_actions, 
            simulated_prefix_next
        )
        
    print("------------------------------------------------")
    print(f"Resulting Intrinsic Reward Shape:       {rewards.shape}  <-- Baked into Offline Q-Targets!")
    print(f"Sample Scaled Rewards:                  {rewards.numpy()}")