# Copyright 2026 starVLA community & Fabian. All rights reserved.
# Licensed under the MIT License, Version 1.0.

import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter

import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
import yaml

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import get_scheduler

# starVLA Modules
from starVLA.dataloader import build_dataloader
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args, build_param_lr_groups
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

# Your Module
from jepa_modules import VisionTransformerPredictorAC

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
logger = get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
    return output_dir

class JEPATrainer(TrainerUtils):
    def __init__(self, cfg, vla_backbone, predictor, dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.vla_backbone = vla_backbone  # Frozen feature extractor
        self.predictor = predictor        # Trainable dynamics transformer
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.run_root_dir, cfg.run_id, "tensorboard"))
        
        self.completed_steps = 0
        self.vision_tokens = 256  # Sliced visual grid from PaliGemma patch count
        
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # 1. ENFORCE STRICT FREEZING OF THE HEAVY VLA FOUNDATION BACKBONE
        self.vla_backbone.eval()
        for param in self.vla_backbone.parameters():
            param.requires_grad = False

        # 2. Predictor is trainable
        self.predictor.train()

        # Print weights being optimized (should only be your predictor modules)
        self.print_trainable_parameters(self.predictor)

        # 3. Register elements into accelerate's execution wrapper
        self.predictor, self.optimizer, self.dataloader = self.setup_distributed_training(
            self.accelerator, self.predictor, self.optimizer, self.dataloader
        )
        
        # Explicitly push frozen backbone to target device
        self.vla_backbone = self.accelerator.prepare_one_model(self.vla_backbone, prepare_target=True)

    def train(self):
        self.dataloader_iter = iter(self.dataloader)
        progress_bar = tqdm(range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process)

        while self.completed_steps < self.config.trainer.max_train_steps:
            try:
                batch = next(self.dataloader_iter)
            except StopIteration:
                if not hasattr(self, "epoch_count"): self.epoch_count = 0
                self.dataloader_iter, self.epoch_count = self._reset_dataloader(self.dataloader, self.epoch_count)
                batch = next(self.dataloader_iter)

            # --- FORWARD & BACKWARD PASS ---
            with self.accelerator.accumulate(self.predictor):
                self.optimizer.zero_grad()
                
                # Unpack trajectory windows from your sequential dataset wrapper
                # obs_t: [B, C, H, W] (Initial Context Frame)
                # actions: [B, T, 8] (Continuous action sequence chunk)
                # obs_targets: [B, T, C, H, W] (Future ground truth images)
                obs_t, actions, obs_targets = batch["obs_t"], batch["actions_horizon"], batch["obs_targets"]
                B, T, _, _, _ = obs_targets.shape

                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        # Extract context embeddings via frozen prefix encoder
                        # starVLA models typically expose an intermediate encoder or forward feature option
                        prefix_out_t = self.vla_backbone.embed_prefix(obs_t)
                        z_t = prefix_out_t[:, :self.vision_tokens, :]

                        # Extract targets across temporal sequence
                        obs_targets_flat = obs_targets.view(B * T, *obs_targets.shape[2:])
                        prefix_out_targets = self.vla_backbone.embed_prefix(obs_targets_flat)
                        
                        y_next = prefix_out_targets[:, :self.vision_tokens, :]
                        y_next = y_next.view(B, T * self.vision_tokens, -1)

                # Trainable Predictor Call
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z_next_pred = self.predictor(z_t, actions)
                    
                    # Compute JEPA Error Matrix
                    loss = torch.nn.functional.mse_loss(z_next_pred, y_next)

                self.accelerator.backward(loss)
                
                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(self.predictor.parameters(), self.config.trainer.gradient_clipping)

                self.optimizer.step()
                self.lr_scheduler.step()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            # Metric Logging & Checkpointing
            if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
                metrics = {
                    "train/jepa_loss": loss.item(),
                    "train/lr": self.lr_scheduler.get_last_lr()[0],
                    "train/epoch": round(self.completed_steps / len(self.dataloader), 2)
                }
                logger.info(f"Step {self.completed_steps} -> JEPA MSE: {loss.item():.6f}")
                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, self.completed_steps)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                if self.accelerator.is_main_process:
                    ckpt_path = os.path.join(self.config.output_dir, "checkpoints", f"jepa_step_{self.completed_steps}.pt")
                    state_dict = self.accelerator.get_state_dict(self.predictor)
                    torch.save(state_dict, ckpt_path)
                    logger.info(f"✅ Predictor checkpoint saved: {ckpt_path}")
                dist.barrier()

        dist.barrier()
        dist.destroy_process_group()

def main(cfg):
    output_dir = setup_directories(cfg)
    
    # Build complete foundation framework structure
    vla_framework = build_framework(cfg)
    
    # Initialize your Action-Conditioned Predictor Architecture natively
    predictor = VisionTransformerPredictorAC(
        num_frames=cfg.jepa.horizon,          # T = 4
        depth=cfg.jepa.depth,                  # Number of Meta block layers
        embed_dim=cfg.jepa.embed_dim,          # 1024 matching your internal sizing
        action_dim=8                           # Franka Bimanual/Relative configurations
    )

    # Build sequential trajectory dataset stream using their registration API
    jepa_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.jepa_data.dataset_py)

    # Setup Optimizer solely tracking your predictor parameters
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=cfg.trainer.learning_rate.base,
        weight_decay=cfg.trainer.optimizer.weight_decay,
        betas=tuple(cfg.trainer.optimizer.betas)
    )

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
    )

    trainer = JEPATrainer(cfg, vla_framework, predictor, jepa_dataloader, optimizer, lr_scheduler, accelerator)
    trainer.prepare_training()
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to your training config")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(clipargs)))
    main(cfg)