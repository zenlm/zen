"""
Training script for Zen-3D model
Handles multi-view, depth, and 3D reconstruction objectives
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
from pathlib import Path
from tqdm import tqdm
import wandb
from dataclasses import dataclass, asdict
import torchvision.transforms as transforms
from PIL import Image
import random

from zen3d_model import Zen3DModel, Zen3DConfig


@dataclass
class TrainingConfig:
    """Training configuration for Zen-3D"""
    # Model settings
    model_name: str = "zen-3d-base"
    checkpoint_dir: str = "./checkpoints/zen3d"

    # Data settings
    data_dir: str = "./data/zoo_3d"
    image_size: int = 224
    max_views: int = 8
    augment_views: bool = True

    # Training settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 100000
    eval_steps: int = 500
    save_steps: int = 2000
    logging_steps: int = 10

    # Loss weights
    language_loss_weight: float = 1.0
    depth_loss_weight: float = 0.5
    coordinate_loss_weight: float = 0.3
    voxel_loss_weight: float = 0.2
    consistency_loss_weight: float = 0.1

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    gradient_checkpointing: bool = True

    # Hardware
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "zen-3d"
    wandb_entity: str = "zoo-labs"


class Zoo3DDataset(Dataset):
    """
    Dataset for Zoo Labs 3D gaming/metaverse scenes
    Includes multi-view images, depth maps, and 3D annotations
    """

    def __init__(self, config: TrainingConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.data_dir = Path(config.data_dir)

        # Load annotations
        self.annotations = self._load_annotations()

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        # Augmentation for training
        if split == "train" and config.augment_views:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomRotation(degrees=10)
            ])
        else:
            self.augment = None

    def _load_annotations(self) -> List[Dict]:
        """Load scene annotations"""
        anno_file = self.data_dir / f"{self.split}_annotations.json"

        # Create sample data if not exists
        if not anno_file.exists():
            os.makedirs(self.data_dir, exist_ok=True)
            sample_data = self._create_sample_annotations()
            with open(anno_file, 'w') as f:
                json.dump(sample_data, f, indent=2)
            return sample_data

        with open(anno_file, 'r') as f:
            return json.load(f)

    def _create_sample_annotations(self) -> List[Dict]:
        """Create sample annotations for demonstration"""
        samples = []
        scene_types = ["marketplace", "arena", "tavern", "dungeon", "castle", "forest"]
        objects = ["sword", "shield", "potion", "chest", "NPC", "monster", "portal", "artifact"]

        for i in range(100):
            sample = {
                "scene_id": f"scene_{i:04d}",
                "scene_type": random.choice(scene_types),
                "description": f"A {random.choice(scene_types)} scene with multiple interactive elements",
                "views": [
                    {
                        "view_id": f"view_{j}",
                        "camera_matrix": np.random.randn(3, 4).tolist(),
                        "view_angles": [
                            j * 45,  # azimuth
                            random.uniform(-30, 30),  # elevation
                            0,  # roll
                            60,  # fov_h
                            45,  # fov_v
                            random.uniform(2, 10)  # distance
                        ]
                    }
                    for j in range(random.randint(3, self.config.max_views))
                ],
                "objects": [
                    {
                        "name": random.choice(objects),
                        "position": [random.uniform(-5, 5) for _ in range(3)],
                        "bbox_3d": [random.uniform(0, 1) for _ in range(6)]
                    }
                    for _ in range(random.randint(2, 8))
                ],
                "depth_available": random.random() > 0.3,
                "voxel_gt_available": random.random() > 0.5
            }
            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample"""
        anno = self.annotations[idx]

        # Load multi-view images (using placeholder data for now)
        images = []
        for view in anno['views']:
            # In production, load actual images
            # img_path = self.data_dir / "images" / anno['scene_id'] / f"{view['view_id']}.jpg"
            # img = Image.open(img_path).convert('RGB')

            # Placeholder: create random image
            img = torch.randn(3, self.config.image_size, self.config.image_size)

            if self.augment and random.random() > 0.5:
                # Apply augmentation (simplified for placeholder)
                img = img + torch.randn_like(img) * 0.1

            images.append(img)

        # Pad to max views
        while len(images) < self.config.max_views:
            images.append(torch.zeros(3, self.config.image_size, self.config.image_size))

        images = torch.stack(images[:self.config.max_views])

        # Camera parameters
        camera_matrices = torch.tensor([
            view['camera_matrix'] + [[0, 0, 0, 0]] * (self.config.max_views - len(anno['views']))
            for view in anno['views']
        ][:self.config.max_views], dtype=torch.float32)

        if camera_matrices.shape[0] < self.config.max_views:
            padding = torch.zeros(self.config.max_views - camera_matrices.shape[0], 3, 4)
            camera_matrices = torch.cat([camera_matrices, padding], dim=0)

        view_angles = torch.tensor([
            view['view_angles'] + [0] * (6 - len(view['view_angles']))
            for view in anno['views']
        ] + [[0] * 6] * (self.config.max_views - len(anno['views'])),
            dtype=torch.float32)[:self.config.max_views]

        # Text description
        text = anno['description']
        # Tokenize (simplified - use actual tokenizer in production)
        input_ids = torch.randint(0, 128256, (256,))  # Placeholder tokens

        # Ground truth for various tasks
        sample = {
            'images': images,
            'camera_matrices': camera_matrices,
            'view_angles': view_angles,
            'input_ids': input_ids,
            'scene_id': anno['scene_id'],
            'num_valid_views': len(anno['views'])
        }

        # Add depth if available
        if anno.get('depth_available'):
            sample['depth_gt'] = torch.randn(self.config.max_views, 1,
                                           self.config.image_size, self.config.image_size)

        # Add voxel ground truth if available
        if anno.get('voxel_gt_available'):
            sample['voxel_gt'] = torch.rand(1, 64, 64, 64) > 0.8

        # Add object coordinates
        if anno.get('objects'):
            coords = torch.tensor([obj['position'] for obj in anno['objects']], dtype=torch.float32)
            sample['coords_gt'] = coords

        return sample


class Zen3DTrainer:
    """Trainer for Zen-3D model"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model
        model_config = Zen3DConfig()
        self.model = Zen3DModel(model_config).to(self.device)

        # Setup data loaders
        self.train_dataset = Zoo3DDataset(config, split="train")
        self.val_dataset = Zoo3DDataset(config, split="val")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1
        )

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=asdict(config),
                name=f"{config.model_name}-{wandb.util.generate_id()}"
            )

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_val_loss = float('inf')

    def compute_losses(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute all training losses"""
        losses = {}

        # Language modeling loss
        if 'logits' in outputs and 'input_ids' in batch:
            # Shift targets for next token prediction
            shift_logits = outputs['logits'][..., :-1, :].contiguous()
            shift_labels = batch['input_ids'][..., 1:].contiguous()

            losses['language'] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        # Depth estimation loss
        if 'depth_maps' in outputs and 'depth_gt' in batch:
            depth_loss = 0
            for i, depth_pred in enumerate(outputs['depth_maps']):
                if i < batch['num_valid_views'].min():
                    depth_loss += F.l1_loss(depth_pred, batch['depth_gt'][:, i])
            losses['depth'] = depth_loss / len(outputs['depth_maps'])

        # Coordinate prediction loss
        if 'coordinates' in outputs and 'coords_gt' in batch:
            # Match predicted coordinates to ground truth (simplified)
            coord_pred = outputs['coordinates'][:, :batch['coords_gt'].shape[1]]
            losses['coordinate'] = F.mse_loss(coord_pred, batch['coords_gt'])

        # Voxel reconstruction loss
        if 'voxels' in outputs and 'voxel_gt' in batch:
            losses['voxel'] = F.binary_cross_entropy(
                outputs['voxels'],
                batch['voxel_gt'].float()
            )

        # Multi-view consistency loss
        if 'multi_view_features' in outputs:
            # Encourage similar features for overlapping regions
            features = outputs['multi_view_features']
            consistency_loss = 0
            for i in range(features.shape[1] - 1):
                consistency_loss += F.mse_loss(
                    features[:, i].mean(dim=1),
                    features[:, i + 1].mean(dim=1)
                )
            losses['consistency'] = consistency_loss / (features.shape[1] - 1)

        return losses

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()}

        # Split images list
        images = [batch['images'][:, i] for i in range(batch['images'].shape[1])]

        # Forward pass with mixed precision
        if self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    images=images,
                    camera_matrices=batch['camera_matrices'],
                    view_angles=batch['view_angles'],
                    input_ids=batch.get('input_ids')
                )
                losses = self.compute_losses(outputs, batch)

                # Weighted total loss
                total_loss = (
                    losses.get('language', 0) * self.config.language_loss_weight +
                    losses.get('depth', 0) * self.config.depth_loss_weight +
                    losses.get('coordinate', 0) * self.config.coordinate_loss_weight +
                    losses.get('voxel', 0) * self.config.voxel_loss_weight +
                    losses.get('consistency', 0) * self.config.consistency_loss_weight
                )
        else:
            outputs = self.model(
                images=images,
                camera_matrices=batch['camera_matrices'],
                view_angles=batch['view_angles'],
                input_ids=batch.get('input_ids')
            )
            losses = self.compute_losses(outputs, batch)

            total_loss = (
                losses.get('language', 0) * self.config.language_loss_weight +
                losses.get('depth', 0) * self.config.depth_loss_weight +
                losses.get('coordinate', 0) * self.config.coordinate_loss_weight +
                losses.get('voxel', 0) * self.config.voxel_loss_weight +
                losses.get('consistency', 0) * self.config.consistency_loss_weight
            )

        # Backward pass
        if self.config.mixed_precision:
            self.scaler.scale(total_loss).backward()
            if self.global_step % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            total_loss.backward()
            if self.global_step % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Update scheduler
        self.scheduler.step()

        # Prepare metrics
        metrics = {
            'loss/total': total_loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
        for name, loss in losses.items():
            if torch.is_tensor(loss):
                metrics[f'loss/{name}'] = loss.item()

        return metrics

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        total_losses = {}
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                images = [batch['images'][:, i] for i in range(batch['images'].shape[1])]

                outputs = self.model(
                    images=images,
                    camera_matrices=batch['camera_matrices'],
                    view_angles=batch['view_angles'],
                    input_ids=batch.get('input_ids')
                )

                losses = self.compute_losses(outputs, batch)

                for name, loss in losses.items():
                    if torch.is_tensor(loss):
                        if name not in total_losses:
                            total_losses[name] = 0
                        total_losses[name] += loss.item()

                num_batches += 1

        # Average losses
        avg_losses = {f'val/{name}': loss / num_batches
                     for name, loss in total_losses.items()}

        return avg_losses

    def save_checkpoint(self, tag: str = None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': asdict(self.config)
        }

        if self.config.mixed_precision and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        tag = tag or f"step_{self.global_step}"
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.max_steps} steps")
        print(f"Device: {self.device}")

        progress_bar = tqdm(total=self.config.max_steps, desc="Training")

        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                # Training step
                metrics = self.train_step(batch)
                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    if self.config.use_wandb:
                        wandb.log(metrics, step=self.global_step)
                    progress_bar.set_postfix(metrics)

                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    val_metrics = self.evaluate()
                    val_loss = sum(val_metrics.values()) / len(val_metrics)

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(tag="best")

                    if self.config.use_wandb:
                        wandb.log(val_metrics, step=self.global_step)

                    print(f"\nValidation at step {self.global_step}:")
                    for name, value in val_metrics.items():
                        print(f"  {name}: {value:.4f}")

                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                progress_bar.update(1)

                if self.global_step >= self.config.max_steps:
                    break

        # Final save
        self.save_checkpoint(tag="final")
        progress_bar.close()
        print("Training completed!")


def main():
    """Main training function"""
    config = TrainingConfig()

    # Override with command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate)
    parser.add_argument("--max_steps", type=int, default=config.max_steps)
    parser.add_argument("--device", type=str, default=config.device)
    args = parser.parse_args()

    # Update config
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.max_steps = args.max_steps
    config.device = args.device

    # Create trainer and start training
    trainer = Zen3DTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()