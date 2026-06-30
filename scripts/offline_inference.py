import argparse
import os
import struct
import time
import cv2
import mmap
import numpy as np
import posix_ipc
import rospy
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from openpi.policies import policy_config
from openpi.training import config as _config
import torch


class OfflineInference:

    def __init__(self):
        # Initialize ROS node first so logs route smoothly
        rospy.init_node("openpi_offline_inference", anonymous=True)

        self.device = "cuda"
        self.dtype = torch.bfloat16

        self.args = self.parse_args()

        self.policy = self.setup_policy()
        self.dataloader = self.load_data()

    def setup_policy(self):
        config = _config.get_config("pi0_kobo_cube_low_mem")
        ckpt_path = os.path.expanduser("~/openpi/checkpoints/pi0_kobo_cube_low_mem/eight_test_cube/20000")
        rospy.loginfo(f"Loading policy from {ckpt_path}")
        policy = policy_config.create_trained_policy(config, ckpt_path)
        rospy.loginfo("Policy loaded successfully.")
        return policy

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Dual-Camera LeRobot v3 VLA Offline Inference Script")
        parser.add_argument(
            "--repo_id", type=str, default="local/bimanual_cube", help="Hugging Face repo or local folder path"
        )
        parser.add_argument(
            "--root",
            type=str,
            default="/home/fabian/lev3_dataset_cube_task_space_orange_external_gripper_shifted",
            help="Root directory of the dataset",
        )
        parser.add_argument(
            "--cam_in_hand", type=str, default="observation.images.cam1", help="In-hand camera stream key"
        )
        parser.add_argument(
            "--cam_external", type=str, default="observation.images.cam2", help="External/Static camera stream key"
        )
        parser.add_argument("--state", type=str, default="observation.state", help="State vector key")

        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--batch_size", type=int, default=1)  # Usually 1 for specific sequential step inference
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--horizon", type=int, default=4, help="Temporal lookahead steps (T)")
        parser.add_argument("--fps", type=int, default=30)
        parser.add_argument("--output_dir", type=str, default="./outputs/jepa_lerobot_dual")
        parser.add_argument("--wandb_project", type=str, default="vla-jepa-curiosity")
        parser.add_argument("--wandb_run_name", type=str, default="dual-cam-jepa-h4")
        return parser.parse_known_args()[0]  # Safe parsing to dodge ROS core arguments

    def load_data(self):
        frame_delta_seconds = 1.0 / self.args.fps

        image_deltas = [i * frame_delta_seconds for i in range(0, self.args.horizon + 1)]
        state_deltas = [i * frame_delta_seconds for i in range(0, self.args.horizon + 1)]
        action_deltas = [i * frame_delta_seconds for i in range(0, self.args.horizon)]

        delta_timestamps = {
            self.args.cam_in_hand: image_deltas,
            self.args.cam_external: image_deltas,
            self.args.state: state_deltas,
            "action": action_deltas,
            }
        dataset = LeRobotDataset(self.args.repo_id, self.args.root, delta_timestamps=delta_timestamps)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
        )

        return dataloader

    def get_vla_latent_predictions(self, example_batch):
        prefix_token, prefix_mask, prefix_ar_mask = self.policy.get_prefix_features(example_batch)
        return prefix_token
    
    def run_inference(self):
        rospy.loginfo("⚡ Starting high-efficiency JEPA latent evaluation loop...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                rospy.loginfo(f"Processing batch {batch_idx + 1}/{len(self.dataloader)}")
                prefix_data, target_future_hand, target_future_ext = self.evaluate_batch(batch)
                if batch_idx % 100 == 0:
                    print(
                        f"[Batch {batch_idx:04d}] "
                        f"Predicted Latents Shape: {prefix_data[0].shape}"
                    )

    def evaluate_batch(self, batch):
        cam_in_hand = batch[self.args.cam_in_hand]    # Shape: [B, T, C, H, W]
        cam_external = batch[self.args.cam_external]  # Shape: [B, T, C, H, W]
        state_tensor = batch[self.args.state]         # Shape: [B, T, D]
        
        batch_size = cam_in_hand.shape[0]
        prefix_data_list = []

        # 1. Loop through each element in the batch to generate token payloads
        for b in range(batch_size):
            img_t0_hand = cam_in_hand[b, 0].permute(1, 2, 0).detach().cpu().numpy()
            img_t0_ext = cam_external[b, 0].permute(1, 2, 0).detach().cpu().numpy()
            current_task_space_state = state_tensor[b, 0].detach().cpu().numpy()

            if img_t0_hand.dtype != np.uint8:
                img_t0_hand = (img_t0_hand * 255.0).astype(np.uint8) if np.max(img_t0_hand) <= 1.0 else img_t0_hand.astype(np.uint8)
            if img_t0_ext.dtype != np.uint8:
                img_t0_ext = (img_t0_ext * 255.0).astype(np.uint8) if np.max(img_t0_ext) <= 1.0 else img_t0_ext.astype(np.uint8)

            obs_payload = {
                "observation/image": img_t0_hand,  
                "observation/external_image": img_t0_ext,
                "observation/state": current_task_space_state,
                "prompt": "pick up the orange cube and place it on the red tape",
                "action": np.zeros((10, 32), dtype=np.float32),
            }

            # Extracts single item features -> List containing [Tokens Tensor]
            single_prefix = self.policy.get_prefix_features(obs_payload)
            
            # Remove the batch dimension added by the inner engine if it outputs [1, 968, 2048]
            token_tensor = single_prefix[0]
            if isinstance(token_tensor, torch.Tensor):
                if token_tensor.dim() == 3 and token_tensor.shape[0] == 1:
                    token_tensor = token_tensor.squeeze(0)
            else:
                if len(token_tensor.shape) == 3 and token_tensor.shape[0] == 1:
                    token_tensor = token_tensor.squeeze(0)
                    
            prefix_data_list.append(token_tensor)

        # 2. Stack tokens back together along the batch axis
        if isinstance(prefix_data_list[0], torch.Tensor):
            prefix_data_combined = torch.stack(prefix_data_list, dim=0) # Shape: [B, 968, 2048]
        else:
            # FIX: Force NumPy to stack/cast to standard float32 first to pass PyTorch's strict type guard
            np_stacked = np.stack(prefix_data_list, axis=0).astype(np.float32)
            prefix_data_combined = torch.from_numpy(np_stacked).to(self.device, dtype=self.dtype)
            
        # Wrap back in a list to preserve the exact interface output expected by train_jepa.py
        prefix_data = [prefix_data_combined]

        # 3. Vectorized tensor extraction for future targets across the entire batch slicing
        target_future_hand = cam_in_hand[:, 1:].to(self.device, dtype=self.dtype)
        target_future_ext = cam_external[:, 1:].to(self.device, dtype=self.dtype)

        return prefix_data, target_future_hand, target_future_ext


if __name__ == "__main__":
    inference_system = OfflineInference()
    try:
        inference_system.run_inference()
    except KeyboardInterrupt:
        rospy.loginfo("Evaluation paused by user.")