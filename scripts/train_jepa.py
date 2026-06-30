# 1. CRITICAL: SILENCE WARNINGS FIRST BEFORE ANY OTHER MODULE LOADS
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.attention")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*sdp_kernel.*")

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import numpy as np
# Explicit imports of your modules
from jepa import VisionTransformerPredictorAC
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Replace with the actual import path of your wrapper class
from offline_inference import OfflineInference

class SequenceResampler(nn.Module):
    def __init__(self, num_queries=512, embed_dim=2048, num_heads=8):
        super().__init__()
        # Learned latents that act as our 512 target slots
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_k = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.attn(self.ln_q(q), self.ln_k(x), x)
        return out

def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Camera LeRobot v3 VLA-JEPA World Model Trainer")
    parser.add_argument("--repo_id", type=str, default="local/bimanual_cube", help="Hugging Face repo or local folder path")
    parser.add_argument("--root", type=str, default="/home/fabian/lev3_dataset_cube_task_space_orange_external_gripper_shifted", help="Root directory of the dataset")
    parser.add_argument("--cam_in_hand", type=str, default="observation.images.cam1", help="In-hand camera stream key")
    parser.add_argument("--cam_external", type=str, default="observation.images.cam2", help="External/Static camera stream key")
    parser.add_argument("--state", type=str, default="observation.state", help="System state feature tracker key")
    
    parser.add_argument("--epochs", type=int, default=50)
    # Set to 4 to stay safely within your VRAM limit while unrolling the VLA policy
    parser.add_argument("--batch_size", type=int, default=4, help="Safe batch size to avoid VRAM OOM errors")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--horizon", type=int, default=4, help="Temporal lookahead steps (T)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="./outputs/jepa_lerobot_dual")
    parser.add_argument("--wandb_project", type=str, default="vla-jepa-curiosity")
    parser.add_argument("--wandb_run_name", type=str, default="dual-cam-jepa-optimized")
    return parser.parse_args()


def train():
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "1000000000000"
    args = parse_args()
    
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    vla_engine = OfflineInference()

    frame_delta_seconds = 1.0 / args.fps
    image_deltas = [i * frame_delta_seconds for i in range(0, args.horizon + 1)] 
    action_deltas = [i * frame_delta_seconds for i in range(0, args.horizon)]     

    delta_timestamps = {
        args.cam_in_hand: image_deltas,   
        args.cam_external: image_deltas,  
        args.state: image_deltas,
        "action": action_deltas           
    }
    
    dataset = LeRobotDataset(args.repo_id, args.root, delta_timestamps=delta_timestamps)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,                  
        drop_last=True, 
        num_workers=4,                 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    predictor = VisionTransformerPredictorAC(
        img_size=(256, 256),
        patch_size=16,
        num_frames=args.horizon,
        embed_dim=2048,             
        predictor_embed_dim=1024,   
        depth=6,
        num_heads=16,
        action_embed_dim=8,
        num_add_tokens=1
    )

    if not hasattr(predictor, "resampler"):
        predictor.resampler = SequenceResampler(num_queries=512, embed_dim=2048).to(device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.MSELoss()

    predictor, optimizer, dataloader = accelerator.prepare(predictor, optimizer, dataloader)
    
    data_iter = iter(dataloader)
    global_step = 0

    # =========================================================
    # OPTIMIZATION TUNING CONFIGS
    # =========================================================
    # Caps the epoch at 2000 steps (~23 min/epoch) to skip redundant overlapping sequences
    MAX_STEPS_PER_EPOCH = 2000 
    # 4 accumulation steps * batch size 4 = Effective batch size of 16
    GRADIENT_ACCUMULATION_STEPS = 4 

    predictor.train()
    progress_bar = tqdm(range(args.epochs), desc="Training JEPA (Epochs)", disable=not accelerator.is_local_main_process)
    start_time = time.time()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        for step_idx in range(MAX_STEPS_PER_EPOCH):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            data_fetch_time = time.time() - start_time
            
            # Extract and bind actions before the forward execution pass
            actions = batch["action"].to(device, dtype=predictor.dtype if hasattr(predictor, "dtype") else torch.bfloat16)
            compute_start = time.time()
            
            with torch.no_grad():
                # =========================================================
                # 1. EXTRACT ANCHOR COGNITIVE CONTEXT (BATCHED PURE VISION SLICE)
                # =========================================================
                prefix_data, tgt_hand, tgt_ext = vla_engine.evaluate_batch(batch)
                t0_tokens = prefix_data[0]
                
                if isinstance(t0_tokens, torch.Tensor):
                    z_context_raw = t0_tokens.to(device, dtype=torch.bfloat16)
                else:
                    z_context_raw = torch.from_numpy(np.array(t0_tokens, dtype=np.float32)).to(device, dtype=torch.bfloat16)

                # Track visual slice over the sequence dimension [B, 512, 2048]
                z_context_jepa = z_context_raw[:, :512, :].clone()

                # =========================================================
                # 2. RUN SEQUENTIAL LOOKAHEAD EXTRACT FOR REAL TARGETS
                # =========================================================
                y_horizon_tokens = []
                for t in range(1, args.horizon + 1):
                    step_batch = {
                        args.cam_in_hand: batch[args.cam_in_hand][:, t:t+1],
                        args.cam_external: batch[args.cam_external][:, t:t+1],
                        args.state: batch[args.state][:, t:t+1],
                    }
                    
                    step_prefix, _, _ = vla_engine.evaluate_batch(step_batch)
                    t_tokens = step_prefix[0]
                    
                    if isinstance(t_tokens, torch.Tensor):
                        t_tensor = t_tokens.to(device, dtype=torch.bfloat16)
                    else:
                        t_tensor = torch.from_numpy(np.array(t_tokens, dtype=np.float32)).to(device, dtype=torch.bfloat16)
                    
                    # Cleanly slice dimension 1, retaining the batch dimensions intact
                    t_vis = t_tensor[:, :512, :].clone() 
                    y_horizon_tokens.append(t_vis)
                
                # Concatenate along the token axis -> [B, 2048, 2048]
                y_next = torch.cat(y_horizon_tokens, dim=1)
                y_next = y_next.to(device, dtype=predictor.dtype if hasattr(predictor, "dtype") else torch.bfloat16)

            # =========================================================
            # 3. FORWARD PREDICTOR PASS & ACCUMULATION STEP
            # =========================================================
            z_next_pred = predictor(z_context_jepa, actions) 
            
            loss = criterion(z_next_pred.float(), y_next.float())        
            scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
            accelerator.backward(scaled_loss)
            
            # Step the optimizer only at the end of our accumulation window
            if (step_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step_idx + 1) == MAX_STEPS_PER_EPOCH:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                    global_step += 1
                    
                optimizer.step()
                optimizer.zero_grad()
            
            compute_time = time.time() - compute_start
            total_step_time = time.time() - start_time
            
            if accelerator.is_main_process and step_idx % 10 == 0:
                wandb.log({"loss/train_step": loss.item(), "training/global_step": global_step, "training/epoch": epoch + 1}, step=global_step)
                if step_idx % 100 == 0:
                    print(f"\n[Epoch {epoch+1}/{args.epochs} | Step {step_idx}/{MAX_STEPS_PER_EPOCH}] Data Wait: {data_fetch_time:.3f}s | GPU Compute: {compute_time:.3f}s")
                    progress_bar.set_postfix({"JEPA MSE": f"{loss.item():.5f}"})
            
            start_time = time.time()
            
        progress_bar.update(1)
        current_epoch = epoch + 1
        if accelerator.is_main_process:
            print(f"\n-> Epoch {current_epoch} Complete.")
            if current_epoch % 5 == 0:
                ckpt_path = os.path.join(args.output_dir, "checkpoints", f"predictor_epoch_{current_epoch}.pt")
                torch.save(accelerator.unwrap_model(predictor).state_dict(), ckpt_path)
                print(f"✅ Checkpoint saved to {ckpt_path}")

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    train()