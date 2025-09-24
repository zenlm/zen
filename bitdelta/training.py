"""
BitDelta training pipeline for personalized model fine-tuning.

Implements BitDelta-aware training with straight-through estimators
and progressive quantization for optimal 1-bit compression.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, List
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

from bitdelta import (
    BitDeltaConfig, BitDeltaModel, BitDeltaEncoder, 
    BitDeltaDecoder, compute_compression_ratio
)


logger = logging.getLogger(__name__)


class BitDeltaTrainer:
    """Trainer for BitDelta personalized fine-tuning."""
    
    def __init__(self,
                 base_model: nn.Module,
                 config: BitDeltaConfig = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize BitDelta trainer.
        
        Args:
            base_model: Pre-trained base model to personalize
            config: BitDelta configuration
            device: Device for training
        """
        self.config = config or BitDeltaConfig()
        self.device = device
        
        # Create BitDelta model
        self.model = BitDeltaModel(base_model, config).to(device)
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.step = 0
        self.best_loss = float('inf')
        
        # Encoder/decoder for checkpointing
        self.encoder = BitDeltaEncoder(config)
        self.decoder = BitDeltaDecoder(config)
    
    def setup_optimizer(self, 
                       learning_rate: float = 1e-4,
                       weight_decay: float = 0.01,
                       warmup_steps: Optional[int] = None):
        """Setup optimizer and scheduler."""
        # Separate parameters for different learning rates
        delta_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'delta' in name:
                delta_params.append(param)
            else:
                other_params.append(param)
        
        # Scale learning rate for delta parameters
        param_groups = [
            {'params': delta_params, 'lr': learning_rate * self.config.learning_rate_scale},
            {'params': other_params, 'lr': learning_rate}
        ]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        
        # Setup warmup scheduler if specified
        if warmup_steps or self.config.warmup_steps:
            warmup = warmup_steps or self.config.warmup_steps
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup
            )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Compute loss (assuming model returns dict with 'loss' key)
        if isinstance(outputs, dict):
            loss = outputs['loss']
        else:
            loss = outputs
        
        # Add BitDelta regularization
        reg_loss = self.model.get_regularization_loss()
        total_loss = loss + reg_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        if self.scheduler and self.step < self.config.warmup_steps:
            self.scheduler.step()
        
        self.step += 1
        
        return {
            'loss': loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                
                if isinstance(outputs, dict):
                    loss = outputs['loss']
                else:
                    loss = outputs
                
                batch_size = next(iter(batch.values())).shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Compute compression ratio
        bitdelta_params = self.model.export_bitdelta()
        compression_ratio = compute_compression_ratio(
            self.model.base_model, bitdelta_params
        )
        
        return {
            'val_loss': avg_loss,
            'compression_ratio': compression_ratio
        }
    
    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              epochs: int = 3,
              learning_rate: float = 1e-4,
              save_path: Optional[str] = None,
              eval_steps: int = 100,
              save_steps: int = 500,
              logging_steps: int = 10):
        """
        Full training loop for BitDelta personalization.
        
        Args:
            train_dataloader: Training data
            val_dataloader: Validation data
            epochs: Number of epochs
            learning_rate: Learning rate
            save_path: Path to save checkpoints
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log metrics every N steps
        """
        # Setup optimizer
        self.setup_optimizer(learning_rate)
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            epoch_losses = []
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for batch in progress_bar:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['total_loss'])
                global_step += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = np.mean(epoch_losses[-logging_steps:])
                    progress_bar.set_postfix({'loss': f"{avg_loss:.4f}"})
                    logger.debug(f"Step {global_step}: {metrics}")
                
                # Evaluation
                if val_dataloader and global_step % eval_steps == 0:
                    eval_metrics = self.evaluate(val_dataloader)
                    logger.info(f"Validation at step {global_step}: {eval_metrics}")
                    
                    if eval_metrics['val_loss'] < best_val_loss:
                        best_val_loss = eval_metrics['val_loss']
                        if save_path:
                            self.save_checkpoint(f"{save_path}/best_model.pt")
                
                # Save checkpoint
                if save_path and global_step % save_steps == 0:
                    self.save_checkpoint(f"{save_path}/checkpoint_{global_step}.pt")
            
            # End of epoch evaluation
            if val_dataloader:
                eval_metrics = self.evaluate(val_dataloader)
                logger.info(f"End of epoch {epoch + 1}: {eval_metrics}")
        
        logger.info("Training completed!")
        
        # Save final model
        if save_path:
            self.save_checkpoint(f"{save_path}/final_model.pt")
    
    def save_checkpoint(self, path: str):
        """Save BitDelta checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export BitDelta parameters
        bitdelta_params = self.model.export_bitdelta()
        
        # Compute compression ratio
        compression_ratio = compute_compression_ratio(
            self.model.base_model, bitdelta_params
        )
        
        checkpoint = {
            'bitdelta_params': bitdelta_params,
            'config': self.config.__dict__,
            'step': self.step,
            'compression_ratio': compression_ratio
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path} (compression ratio: {compression_ratio:.1f}x)")
    
    def load_checkpoint(self, path: str):
        """Load BitDelta checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load config
        self.config = BitDeltaConfig(**checkpoint['config'])
        
        # Load BitDelta parameters
        bitdelta_params = checkpoint['bitdelta_params']
        
        # Apply to model
        for name, module in self.model.named_modules():
            if hasattr(module, 'delta_signs'):
                param_name = f"{name}.weight"
                if param_name in bitdelta_params:
                    signs, scales = bitdelta_params[param_name]
                    module.delta_signs.data = signs.to(self.device)
                    module.delta_scales.data = scales.to(self.device)
        
        self.step = checkpoint.get('step', 0)
        logger.info(f"Loaded checkpoint from {path}")


class ProgressiveQuantizationTrainer(BitDeltaTrainer):
    """
    Advanced trainer with progressive quantization.
    
    Gradually transitions from full precision to 1-bit during training
    for better convergence and quality.
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 config: BitDeltaConfig = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 quantization_schedule: Optional[List[tuple]] = None):
        """
        Initialize progressive quantization trainer.
        
        Args:
            base_model: Base model to personalize
            config: BitDelta configuration
            device: Training device
            quantization_schedule: List of (step, quantization_level) tuples
        """
        super().__init__(base_model, config, device)
        
        # Default schedule: gradually increase quantization
        self.quantization_schedule = quantization_schedule or [
            (0, 0.0),      # Start with no quantization
            (500, 0.25),   # 25% quantization
            (1000, 0.5),   # 50% quantization
            (1500, 0.75),  # 75% quantization
            (2000, 1.0),   # Full 1-bit quantization
        ]
        
        self.current_quantization = 0.0
    
    def get_quantization_level(self) -> float:
        """Get current quantization level based on training step."""
        for step, level in reversed(self.quantization_schedule):
            if self.step >= step:
                return level
        return 0.0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with progressive quantization."""
        # Update quantization level
        self.current_quantization = self.get_quantization_level()
        
        # Apply quantization level to model
        for module in self.model.modules():
            if hasattr(module, 'delta_signs'):
                # Mix between full precision and quantized
                if self.current_quantization < 1.0:
                    # Interpolate between full and quantized
                    full_deltas = module.delta_signs
                    quantized = torch.sign(module.delta_signs)
                    module.delta_signs.data = (
                        (1 - self.current_quantization) * full_deltas +
                        self.current_quantization * quantized
                    )
        
        # Regular training step
        metrics = super().train_step(batch)
        metrics['quantization_level'] = self.current_quantization
        
        return metrics


class MultiProfileTrainer:
    """
    Trainer for managing multiple BitDelta profiles.
    
    Enables training and switching between multiple personalization
    profiles from a single base model.
    """
    
    def __init__(self,
                 base_model: nn.Module,
                 config: BitDeltaConfig = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize multi-profile trainer.
        
        Args:
            base_model: Shared base model
            config: BitDelta configuration
            device: Training device
        """
        self.base_model = base_model
        self.config = config or BitDeltaConfig()
        self.device = device
        
        # Store multiple profiles
        self.profiles: Dict[str, Dict[str, tuple]] = {}
        self.active_profile: Optional[str] = None
        
        # Encoder/decoder
        self.encoder = BitDeltaEncoder(config)
        self.decoder = BitDeltaDecoder(config)
    
    def create_profile(self, profile_name: str):
        """Create a new personalization profile."""
        if profile_name in self.profiles:
            logger.warning(f"Profile {profile_name} already exists")
            return
        
        # Initialize empty profile
        self.profiles[profile_name] = {}
        self.active_profile = profile_name
        logger.info(f"Created profile: {profile_name}")
    
    def train_profile(self,
                     profile_name: str,
                     train_dataloader: DataLoader,
                     val_dataloader: Optional[DataLoader] = None,
                     **kwargs):
        """Train a specific profile."""
        if profile_name not in self.profiles:
            self.create_profile(profile_name)
        
        self.active_profile = profile_name
        
        # Create trainer for this profile
        trainer = BitDeltaTrainer(self.base_model, self.config, self.device)
        
        # Train
        trainer.train(train_dataloader, val_dataloader, **kwargs)
        
        # Store trained profile
        self.profiles[profile_name] = trainer.model.export_bitdelta()
        
        logger.info(f"Trained profile: {profile_name}")
    
    def switch_profile(self, profile_name: str) -> nn.Module:
        """Switch to a different profile and return personalized model."""
        if profile_name not in self.profiles:
            raise ValueError(f"Profile {profile_name} not found")
        
        self.active_profile = profile_name
        
        # Get base state
        base_state = self.base_model.state_dict()
        
        # Apply BitDelta profile
        personalized_state = self.decoder.decode_model(
            base_state, self.profiles[profile_name]
        )
        
        # Create personalized model
        personalized_model = type(self.base_model)()
        personalized_model.load_state_dict(personalized_state)
        
        return personalized_model
    
    def merge_profiles(self, 
                       profile_names: List[str],
                       weights: Optional[List[float]] = None,
                       new_profile_name: str = 'merged') -> Dict[str, tuple]:
        """
        Merge multiple profiles with optional weighting.
        
        Args:
            profile_names: Profiles to merge
            weights: Optional weights for each profile
            new_profile_name: Name for merged profile
            
        Returns:
            Merged BitDelta parameters
        """
        if weights is None:
            weights = [1.0 / len(profile_names)] * len(profile_names)
        
        merged = {}
        
        # Get all parameter names
        all_params = set()
        for name in profile_names:
            all_params.update(self.profiles[name].keys())
        
        # Merge each parameter
        for param_name in all_params:
            signs_list = []
            scales_list = []
            valid_weights = []
            
            for i, profile_name in enumerate(profile_names):
                if param_name in self.profiles[profile_name]:
                    signs, scales = self.profiles[profile_name][param_name]
                    signs_list.append(signs)
                    scales_list.append(scales)
                    valid_weights.append(weights[i])
            
            if signs_list:
                # Normalize weights
                valid_weights = np.array(valid_weights)
                valid_weights /= valid_weights.sum()
                
                # Weighted average of signs (then re-quantize)
                merged_signs = sum(w * s for w, s in zip(valid_weights, signs_list))
                merged_signs = torch.sign(merged_signs)
                
                # Weighted average of scales
                merged_scales = sum(w * s for w, s in zip(valid_weights, scales_list))
                
                merged[param_name] = (merged_signs, merged_scales)
        
        # Store merged profile
        self.profiles[new_profile_name] = merged
        
        return merged
    
    def save_profiles(self, path: str):
        """Save all profiles to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for profile_name, profile_data in self.profiles.items():
            profile_path = path / f"{profile_name}.pt"
            
            # Compute compression ratio
            compression_ratio = compute_compression_ratio(
                self.base_model, profile_data
            )
            
            checkpoint = {
                'profile_name': profile_name,
                'bitdelta_params': profile_data,
                'config': self.config.__dict__,
                'compression_ratio': compression_ratio
            }
            
            torch.save(checkpoint, profile_path)
            logger.info(f"Saved profile {profile_name} to {profile_path}")
    
    def load_profiles(self, path: str):
        """Load profiles from disk."""
        path = Path(path)
        
        for profile_path in path.glob("*.pt"):
            checkpoint = torch.load(profile_path, map_location=self.device)
            profile_name = checkpoint['profile_name']
            self.profiles[profile_name] = checkpoint['bitdelta_params']
            logger.info(f"Loaded profile {profile_name} from {profile_path}")