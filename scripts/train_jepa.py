import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import wandb

# Explicit imports of your modules
from jepa import VisionTransformerPredictorAC
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Camera LeRobot v3 VLA-JEPA World Model Trainer")
    parser.add_argument("--repo_id", type=str, default="local/bimanual_cube", help="Hugging Face repo or local folder path")
    parser.add_argument("--root", type=str, default="/home/fabian/lev3_dataset_cube_task_space_orange_external_gripper_shifted", help="Root directory of the dataset")
    # Dual camera key properties matching your v3 schema configurations
    parser.add_argument("--cam_in_hand", type=str, default="observation.images.cam1", help="In-hand camera stream key")
    parser.add_argument("--cam_external", type=str, default="observation.images.cam2", help="External/Static camera stream key")
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--horizon", type=int, default=4, help="Temporal lookahead steps (T)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="./outputs/jepa_lerobot_dual")
    parser.add_argument("--wandb_project", type=str, default="vla-jepa-curiosity")
    parser.add_argument("--wandb_run_name", type=str, default="dual-cam-jepa-h4")
    return parser.parse_args()


def mock_vla_prefix_embed(x):
    """
    Mock placeholder mirroring your frozen VLA patch token extraction.
    Input: [B, C, H, W] -> Output: Latent patches [B, 256, 1152]
    """
    B = x.shape[0]
    return torch.zeros(B, 256, 1152, device=x.device, dtype=x.dtype)


def train():
    # Force environmental virtual RAM mapping hints for Hugging Face datasets backend
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "1000000000000"
    
    args = parse_args()
    
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Compute step time-deltas
    frame_delta_seconds = 1.0 / args.fps
    
    # FIX: Include 0.0 so the current anchor frame (index 0) is natively fetched inside the cache chunk
    image_deltas = [i * frame_delta_seconds for i in range(0, args.horizon + 1)] # Length: 5 (t0, t1, t2, t3, t4)
    action_deltas = [i * frame_delta_seconds for i in range(0, args.horizon)]     # Length: 4 (t0, t1, t2, t3)

    delta_timestamps = {
        args.cam_in_hand: image_deltas,   # Yields [B, horizon + 1, C, H, W]
        args.cam_external: image_deltas,  # Yields [B, horizon + 1, C, H, W]
        "action": action_deltas           # Yields [B, horizon, action_dim]
    }
    
    dataset = LeRobotDataset(args.repo_id, args.root, delta_timestamps=delta_timestamps)
    
    # High-Performance Loader optimized for Parquet chunks
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,                  # Safe to shuffle now that files are memory-mapped
        drop_last=True, 
        num_workers=4,                 # Prevents multi-process disk thrashing on Parquet formats
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Spatial token aggregation sizing update
    vision_tokens_per_cam = 256
    total_spatial_tokens = vision_tokens_per_cam * 2 # 512 total context features

    predictor = VisionTransformerPredictorAC(
        img_size=(256, 256),
        patch_size=16,
        num_frames=args.horizon,
        embed_dim=1152,            # Explicitly match OpenPI/PaliGemma channels
        predictor_embed_dim=1024,   # Internal JEPA transformer hidden layout
        depth=6,
        num_heads=16,
        action_embed_dim=8,
        num_add_tokens=1
    )

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.MSELoss()

    predictor, optimizer, dataloader = accelerator.prepare(predictor, optimizer, dataloader)
    
    # --- STARVLA DECOUPLED DECOUPLING: INITIALIZE PERSISTENT ITERATOR ---
    data_iter = iter(dataloader)
    steps_per_epoch = len(dataloader)
    total_training_steps = args.epochs * steps_per_epoch
    global_step = 0

    if accelerator.is_main_process:
        print(f"🧠 Persistent data streaming stream active. Running {total_training_steps} total steps.")

    predictor.train()
    progress_bar = tqdm(range(total_training_steps), desc="Training JEPA", disable=not accelerator.is_local_main_process)
    start_time = time.time()

    while global_step < total_training_steps:
        # Fetch data seamlessly using persistent state calls to next()
        try:
            batch = next(data_iter)
        except StopIteration:
            if accelerator.is_main_process:
                print(f"\n🔄 Epoch boundary reached around step {global_step}. Refreshing state iterator hooks...")
            data_iter = iter(dataloader)
            batch = next(data_iter)

        data_fetch_time = time.time() - start_time
        optimizer.zero_grad()
        
        # 1. Pull current anchor frames (t=0 index natively fetched inside memory maps)
        img_hand_t0 = batch[args.cam_in_hand][:, 0]
        img_ext_t0 = batch[args.cam_external][:, 0]
        
        # 2. Slice future lookahead trajectories safely (Index 1 to end -> t1, t2, t3, t4)
        targets_hand = batch[args.cam_in_hand][:, 1:]
        targets_ext = batch[args.cam_external][:, 1:]
        actions = batch["action"]
        
        B, T, C_img, H_img, W_img = targets_hand.shape
        compute_start = time.time()
        
        with torch.no_grad():
            # Extract initial context latents across both viewpoints
            z_hand_t0 = mock_vla_prefix_embed(img_hand_t0)[:, :vision_tokens_per_cam, :]
            z_ext_t0 = mock_vla_prefix_embed(img_ext_t0)[:, :vision_tokens_per_cam, :]
            # Concatenate along the sequence axis -> [B, 512, 1152]
            z_context = torch.cat([z_hand_t0, z_ext_t0], dim=1)

            # FIX: Reshape non-contiguous memory slices cleanly instead of crashing view size properties
            flat_hand = targets_hand.reshape(B * T, C_img, H_img, W_img)
            flat_ext = targets_ext.reshape(B * T, C_img, H_img, W_img)
            
            y_hand_all = mock_vla_prefix_embed(flat_hand)[:, :vision_tokens_per_cam, :]
            y_ext_all = mock_vla_prefix_embed(flat_ext)[:, :vision_tokens_per_cam, :]
            
            # Reconstruct multi-view target sequence
            y_combined = torch.cat([y_hand_all, y_ext_all], dim=1) 
            y_next = y_combined.view(B, T * total_spatial_tokens, -1)

        # Pass unified spatial context matrix to predictor
        z_next_pred = predictor(z_context, actions) 
        loss = criterion(z_next_pred, y_next)
        
        accelerator.backward(loss)
        
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            progress_bar.update(1)
            global_step += 1
            
        optimizer.step()
        
        compute_time = time.time() - compute_start
        total_step_time = time.time() - start_time
        
        # Performance Evaluation Metrics Logging
        if accelerator.is_main_process and global_step % 10 == 0:
            wandb.log({"loss/train_step": loss.item(), "training/global_step": global_step}, step=global_step)
            print(f"\n[Step {global_step}] Data Wait: {data_fetch_time:.3f}s | GPU Compute: {compute_time:.3f}s | Total: {total_step_time:.3f}s")
            progress_bar.set_postfix({"JEPA MSE": f"{loss.item():.5f}"})

        # Checkpoint Management
        if global_step % steps_per_epoch == 0 and global_step > 0:
            current_epoch = global_step // steps_per_epoch
            if accelerator.is_main_process:
                print(f"-> Epoch {current_epoch} Complete.")
                if current_epoch % 10 == 0:
                    ckpt_path = os.path.join(args.output_dir, "checkpoints", f"predictor_epoch_{current_epoch}.pt")
                    torch.save(accelerator.unwrap_model(predictor).state_dict(), ckpt_path)
                    print(f"✅ Checkpoint saved to {ckpt_path}")
        
        # Reset tick timer for accurate Data Wait metrics
        start_time = time.time()

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    train()