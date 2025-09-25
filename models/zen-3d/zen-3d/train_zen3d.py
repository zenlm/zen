"""
Training script for Zen-3D model
Multi-view spatial understanding with Zoo Labs integration
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
from tqdm import tqdm
import wandb
from dataclasses import dataclass, field
from pathlib import Path

from zen_3d_model import Zen3DModel, Zen3DConfig
from zoo_integration import ZooZen3DIntegration, ZooMetaverseConfig


@dataclass
class TrainingConfig:
    """Training configuration for Zen-3D"""

    # Model settings
    base_model: str = "HanzoLabs/zen-omni-30b"
    vision_model: str = "openai/clip-vit-large-patch14-336"

    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 1000
    max_steps: int = 100000
    eval_steps: int = 500
    save_steps: int = 2000
    logging_steps: int = 100

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    gradient_checkpointing: bool = True

    # Data settings
    max_views: int = 8
    image_size: int = 336
    max_seq_length: int = 2048

    # Paths
    data_dir: str = "./data/multiview"
    output_dir: str = "./models/zen-3d"
    cache_dir: str = "./cache"

    # Task weights for multi-task learning
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        "reconstruction": 1.0,
        "depth": 0.5,
        "scene_graph": 0.8,
        "nft": 0.3,
        "physics": 0.3,
        "avatar": 0.3
    })

    # W&B settings
    use_wandb: bool = True
    wandb_project: str = "zen-3d"
    wandb_entity: str = "zoo-labs"


class MultiViewDataset(Dataset):
    """Dataset for multi-view 3D understanding"""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_views: int = 8,
        image_size: int = 336
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_views = max_views
        self.image_size = image_size

        # Load annotations
        self.annotations = self._load_annotations()

        # Initialize transforms
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations"""
        anno_path = self.data_dir / f"{self.split}_annotations.json"

        if not anno_path.exists():
            # Create dummy annotations for testing
            return self._create_dummy_annotations()

        with open(anno_path) as f:
            return json.load(f)

    def _create_dummy_annotations(self) -> List[Dict]:
        """Create dummy annotations for testing"""
        annotations = []
        for i in range(100):
            anno = {
                "id": f"{self.split}_{i:05d}",
                "views": [
                    {
                        "image_path": f"view_{v}.jpg",
                        "camera_position": [
                            np.cos(v * np.pi / 4),
                            np.sin(v * np.pi / 4),
                            1.0
                        ],
                        "camera_angle": [v * 45, 0]
                    }
                    for v in range(min(4, self.max_views))
                ],
                "description": f"A 3D scene with multiple objects viewed from {min(4, self.max_views)} angles",
                "objects": [
                    {
                        "name": f"object_{j}",
                        "position": [np.random.randn() * 2 for _ in range(3)],
                        "category": np.random.choice(["furniture", "vehicle", "character"])
                    }
                    for j in range(np.random.randint(3, 8))
                ],
                "task_data": {
                    "nft": {
                        "category": "character",
                        "rarity": np.random.rand(),
                        "traits": [f"trait_{k}" for k in range(5)]
                    },
                    "physics": {
                        "mass": np.random.rand() * 100,
                        "velocity": [np.random.randn() for _ in range(3)]
                    }
                }
            }
            annotations.append(anno)
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        anno = self.annotations[idx]

        # Load images (using dummy data for now)
        images = []
        positions = []
        angles = []

        for view in anno["views"][:self.max_views]:
            # In real implementation, load actual images
            # img = Image.open(self.data_dir / anno["id"] / view["image_path"])
            # img = self.transform(img)

            # Dummy image for testing
            img = torch.randn(3, self.image_size, self.image_size)
            images.append(img)

            positions.append(view["camera_position"])
            angles.append(view["camera_angle"])

        # Pad if fewer views than max
        while len(images) < self.max_views:
            images.append(torch.zeros(3, self.image_size, self.image_size))
            positions.append([0, 0, 0])
            angles.append([0, 0])

        return {
            "images": torch.stack(images),
            "positions": torch.tensor(positions, dtype=torch.float32),
            "angles": torch.tensor(angles, dtype=torch.float32),
            "description": anno["description"],
            "objects": anno["objects"],
            "task_data": anno.get("task_data", {}),
            "num_valid_views": len(anno["views"])
        }


class Zen3DTrainer:
    """Trainer for Zen-3D model"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = self._init_model()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)

        # Initialize datasets
        self.train_dataset = MultiViewDataset(
            config.data_dir, "train",
            config.max_views, config.image_size
        )
        self.val_dataset = MultiViewDataset(
            config.data_dir, "val",
            config.max_views, config.image_size
        )

        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Initialize optimizer and scheduler
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.__dict__
            )

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _init_model(self) -> ZooZen3DIntegration:
        """Initialize model with Zoo Labs integration"""
        zen3d_config = Zen3DConfig(
            base_model=self.config.base_model,
            vision_model=self.config.vision_model,
            max_views=self.config.max_views,
            gradient_checkpointing=self.config.gradient_checkpointing
        )

        zen3d_model = Zen3DModel(zen3d_config)
        zoo_config = ZooMetaverseConfig()
        model = ZooZen3DIntegration(zen3d_model, zoo_config)

        if self.config.fp16:
            model = model.half()

        model = model.to(self.device)

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model

    def _init_optimizer(self):
        """Initialize optimizer"""
        # Separate parameters by type
        no_decay = ["bias", "LayerNorm.weight", "ln"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        return optimizer

    def _init_scheduler(self):
        """Initialize learning rate scheduler"""
        num_training_steps = self.config.max_steps
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        return scheduler

    def compute_losses(self, batch: Dict, outputs: Dict) -> Dict[str, torch.Tensor]:
        """Compute multi-task losses"""
        losses = {}

        # Reconstruction loss (main language modeling)
        if "logits" in outputs["base_outputs"]:
            labels = self.tokenizer(
                batch["description"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length
            ).input_ids.to(self.device)

            reconstruction_loss = nn.CrossEntropyLoss()(
                outputs["base_outputs"]["logits"].reshape(-1, outputs["base_outputs"]["logits"].shape[-1]),
                labels.reshape(-1)
            )
            losses["reconstruction"] = reconstruction_loss

        # Depth estimation loss
        if "depth_maps" in outputs["base_outputs"] and outputs["base_outputs"]["depth_maps"] is not None:
            # Would compare with ground truth depth
            depth_loss = torch.mean(outputs["base_outputs"]["depth_maps"])  # Placeholder
            losses["depth"] = depth_loss

        # Scene graph loss
        if "scene_graph" in outputs["base_outputs"]:
            # Compare predicted coordinates with ground truth
            scene_graph_loss = torch.mean(
                outputs["base_outputs"]["scene_graph"]["object_coords"] ** 2
            )  # Placeholder
            losses["scene_graph"] = scene_graph_loss

        # NFT classification loss
        if "nft" in outputs and "category_logits" in outputs["nft"]:
            # Would compare with ground truth NFT categories
            nft_loss = torch.mean(outputs["nft"]["category_logits"])  # Placeholder
            losses["nft"] = nft_loss

        # Physics prediction loss
        if "physics" in outputs:
            physics_loss = torch.mean(outputs["physics"]["velocities"] ** 2)  # Placeholder
            losses["physics"] = physics_loss

        # Avatar keypoint loss
        if "avatar" in outputs and outputs["avatar"]["keypoints"] is not None:
            avatar_loss = torch.mean(outputs["avatar"]["keypoints"] ** 2)  # Placeholder
            losses["avatar"] = avatar_loss

        # Compute weighted total loss
        total_loss = sum(
            self.config.task_weights.get(name, 1.0) * loss
            for name, loss in losses.items()
        )
        losses["total"] = total_loss

        return losses

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        # Move batch to device
        images = batch["images"].to(self.device)
        positions = batch["positions"].to(self.device)
        angles = batch["angles"].to(self.device)

        # Split images into list of views
        images_list = [images[:, i] for i in range(images.shape[1])]

        # Forward pass
        outputs = self.model(
            images_list,
            coordinates=positions,
            angles=angles,
            task="full"
        )

        # Compute losses
        losses = self.compute_losses(batch, outputs)

        # Backward pass
        loss = losses["total"] / self.config.gradient_accumulation_steps
        loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Convert losses to float for logging
        loss_dict = {k: v.item() for k, v in losses.items()}

        return loss_dict

    @torch.no_grad()
    def eval_step(self, batch: Dict) -> Dict[str, float]:
        """Single evaluation step"""
        self.model.eval()

        # Move batch to device
        images = batch["images"].to(self.device)
        positions = batch["positions"].to(self.device)
        angles = batch["angles"].to(self.device)

        # Split images into list of views
        images_list = [images[:, i] for i in range(images.shape[1])]

        # Forward pass
        outputs = self.model(
            images_list,
            coordinates=positions,
            angles=angles,
            task="full"
        )

        # Compute losses
        losses = self.compute_losses(batch, outputs)

        # Convert to float
        loss_dict = {k: v.item() for k, v in losses.items()}

        return loss_dict

    def evaluate(self) -> Dict[str, float]:
        """Full evaluation loop"""
        self.model.eval()
        total_losses = {}
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                losses = self.eval_step(batch)

                for k, v in losses.items():
                    if k not in total_losses:
                        total_losses[k] = 0
                    total_losses[k] += v

                num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        return avg_losses

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Loaded checkpoint from {path}")

    def train(self):
        """Main training loop"""
        print("Starting training...")

        for epoch in range(100):  # Large number, will stop at max_steps
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                # Training step
                losses = self.train_step(batch)
                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    if self.config.use_wandb:
                        wandb.log(
                            {"train/" + k: v for k, v in losses.items()},
                            step=self.global_step
                        )
                    print(f"Step {self.global_step}: Loss = {losses['total']:.4f}")

                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    val_losses = self.evaluate()

                    if self.config.use_wandb:
                        wandb.log(
                            {"val/" + k: v for k, v in val_losses.items()},
                            step=self.global_step
                        )

                    print(f"Validation: Loss = {val_losses['total']:.4f}")

                    # Save best model
                    if val_losses["total"] < self.best_val_loss:
                        self.best_val_loss = val_losses["total"]
                        self.save_checkpoint(
                            os.path.join(self.config.output_dir, "best_model.pt")
                        )

                # Regular checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(
                        os.path.join(
                            self.config.output_dir,
                            f"checkpoint_{self.global_step}.pt"
                        )
                    )

                # Stop at max steps
                if self.global_step >= self.config.max_steps:
                    print(f"Reached max steps ({self.config.max_steps})")
                    return


def main():
    """Main training script"""
    config = TrainingConfig(
        data_dir="./data/zoo_multiview",
        output_dir="./models/zen-3d-zoo",
        batch_size=2,  # Small for testing
        gradient_accumulation_steps=16,
        max_steps=10000,
        use_wandb=False  # Set to True for real training
    )

    trainer = Zen3DTrainer(config)

    # Option to resume from checkpoint
    # trainer.load_checkpoint("./models/zen-3d-zoo/checkpoint_5000.pt")

    trainer.train()


if __name__ == "__main__":
    main()