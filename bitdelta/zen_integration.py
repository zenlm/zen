"""
Integration of BitDelta with Zen model architectures.

Provides BitDelta support for all Zen model variants including
zen-nano, zen-omni, and zen-coder.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    Qwen2Config, Qwen2ForCausalLM
)
from typing import Dict, Optional, List, Tuple
import json
from pathlib import Path
import logging

from bitdelta import (
    BitDeltaConfig, BitDeltaEncoder, BitDeltaDecoder,
    BitDeltaLayer, compute_compression_ratio
)


logger = logging.getLogger(__name__)


class ZenBitDelta:
    """BitDelta integration for Zen models."""
    
    SUPPORTED_MODELS = {
        # Zen-nano variants (4B params)
        'zen-nano-4b': 'Qwen/Qwen2.5-3B',
        'zen-nano-chat': 'Qwen/Qwen2.5-3B-Instruct',
        'zen-nano-coder': 'Qwen/Qwen2.5-Coder-3B',
        
        # Zen-omni variants (30B params)
        'zen-omni-30b': 'Qwen/Qwen2.5-32B',
        'zen-omni-chat': 'Qwen/Qwen2.5-32B-Instruct',
        'zen-omni-coder': 'Qwen/Qwen2.5-Coder-32B',
        
        # Specialized models
        'zen-next-7b': 'Qwen/Qwen2.5-7B',
        'zen-coder-14b': 'Qwen/Qwen2.5-Coder-14B',
    }
    
    def __init__(self, 
                 model_name: str,
                 config: Optional[BitDeltaConfig] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize BitDelta for a Zen model.
        
        Args:
            model_name: Name of Zen model variant
            config: BitDelta configuration
            device: Device for model
        """
        self.model_name = model_name
        self.config = config or self._get_optimal_config(model_name)
        self.device = device
        
        # Get base model path
        if model_name in self.SUPPORTED_MODELS:
            self.base_model_path = self.SUPPORTED_MODELS[model_name]
        else:
            self.base_model_path = model_name  # Allow custom paths
        
        # Load base model and tokenizer
        self.tokenizer = None
        self.base_model = None
        self.encoder = BitDeltaEncoder(self.config)
        self.decoder = BitDeltaDecoder(self.config)
        
        # Profile storage
        self.profiles: Dict[str, Dict] = {}
    
    def _get_optimal_config(self, model_name: str) -> BitDeltaConfig:
        """Get optimal BitDelta config for model size."""
        config = BitDeltaConfig()
        
        # Adjust config based on model size
        if 'nano' in model_name or '3B' in model_name or '4B' in model_name:
            # Smaller models: more aggressive compression
            config.block_size = 64
            config.use_per_channel_scale = True
            config.delta_regularization = 0.01
            config.sparsity_penalty = 0.001
            
        elif 'omni' in model_name or '30B' in model_name or '32B' in model_name:
            # Large models: balance quality and compression
            config.block_size = 128
            config.use_per_channel_scale = True
            config.delta_regularization = 0.005
            config.sparsity_penalty = 0.0005
            
        else:
            # Medium models: default settings
            config.block_size = 96
            config.use_per_channel_scale = True
            config.delta_regularization = 0.008
            config.sparsity_penalty = 0.0008
        
        return config
    
    def load_base_model(self):
        """Load the base Zen model."""
        logger.info(f"Loading base model: {self.base_model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        
        # Load model with appropriate dtype for size
        if '30B' in self.model_name or '32B' in self.model_name:
            # Large models: use lower precision
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch_dtype,
            device_map='auto' if self.device == 'cuda' else None
        )
        
        if self.device != 'cuda':
            self.base_model = self.base_model.to(self.device)
        
        logger.info(f"Loaded model with {sum(p.numel() for p in self.base_model.parameters())/1e9:.1f}B parameters")
    
    def apply_bitdelta_layers(self, target_modules: Optional[List[str]] = None):
        """
        Replace model layers with BitDelta versions.
        
        Args:
            target_modules: Specific module names to replace.
                           If None, replaces all linear layers.
        """
        if target_modules is None:
            # Default: target attention and MLP layers
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'gate_proj', 'up_proj', 'down_proj']
        
        replaced_count = 0
        
        for name, module in self.base_model.named_modules():
            # Check if module name matches target patterns
            should_replace = False
            for target in target_modules:
                if target in name:
                    should_replace = True
                    break
            
            if should_replace and isinstance(module, nn.Linear):
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                
                if parent_name:
                    parent = self.base_model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                else:
                    parent = self.base_model
                
                # Replace with BitDelta layer
                bitdelta_layer = BitDeltaLayer(module, self.config)
                setattr(parent, module_name, bitdelta_layer)
                replaced_count += 1
        
        logger.info(f"Replaced {replaced_count} layers with BitDelta versions")
    
    def personalize(self,
                   profile_name: str,
                   train_data: List[Dict],
                   learning_rate: float = 5e-5,
                   epochs: int = 3,
                   batch_size: int = 4):
        """
        Fine-tune model with BitDelta for personalization.
        
        Args:
            profile_name: Name for this personalization profile
            train_data: Training examples
            learning_rate: Learning rate
            epochs: Number of epochs
            batch_size: Batch size
        """
        if self.base_model is None:
            self.load_base_model()
        
        # Apply BitDelta layers if not already done
        if not any(isinstance(m, BitDeltaLayer) for m in self.base_model.modules()):
            self.apply_bitdelta_layers()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=learning_rate
        )
        
        # Training loop
        self.base_model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    [ex['text'] for ex in batch],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Forward pass
                outputs = self.base_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Add BitDelta regularization
                reg_loss = 0
                for module in self.base_model.modules():
                    if isinstance(module, BitDeltaLayer):
                        reg_loss += module.regularization_loss()
                
                total_loss = loss + reg_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")
        
        # Export and save profile
        self._save_profile(profile_name)
    
    def _save_profile(self, profile_name: str):
        """Extract and save BitDelta profile."""
        profile = {
            'model_name': self.model_name,
            'config': self.config.__dict__,
            'deltas': {}
        }
        
        # Extract BitDelta parameters
        for name, module in self.base_model.named_modules():
            if isinstance(module, BitDeltaLayer):
                signs = torch.sign(module.delta_signs)
                scales = module.delta_scales
                profile['deltas'][name] = (signs.cpu(), scales.cpu())
        
        # Compute compression ratio
        base_size = sum(p.numel() * 4 for p in self.base_model.parameters())
        delta_size = sum(
            s.numel() / 8 + sc.numel() * 4 
            for s, sc in profile['deltas'].values()
        )
        profile['compression_ratio'] = base_size / delta_size
        
        self.profiles[profile_name] = profile
        logger.info(f"Saved profile '{profile_name}' with {profile['compression_ratio']:.1f}x compression")
    
    def load_profile(self, profile_name: str) -> nn.Module:
        """
        Load a personalization profile and return personalized model.
        
        Args:
            profile_name: Name of profile to load
            
        Returns:
            Personalized model
        """
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        if self.base_model is None:
            self.load_base_model()
        
        profile = self.profiles[profile_name]
        
        # Apply deltas to base model
        base_state = self.base_model.state_dict()
        personalized_state = self.decoder.decode_model(
            base_state, profile['deltas']
        )
        
        # Create personalized model
        personalized_model = type(self.base_model).from_pretrained(
            self.base_model_path,
            state_dict=personalized_state,
            torch_dtype=self.base_model.dtype
        ).to(self.device)
        
        return personalized_model
    
    def save_to_disk(self, save_dir: str):
        """Save all profiles to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'base_model_path': self.base_model_path,
            'config': self.config.__dict__,
            'profiles': list(self.profiles.keys())
        }
        
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save each profile
        for profile_name, profile_data in self.profiles.items():
            torch.save(profile_data, save_path / f'{profile_name}.pt')
            
            # Also save in compressed format
            compressed_path = save_path / f'{profile_name}.bitdelta'
            self._save_compressed_profile(profile_data, compressed_path)
        
        logger.info(f"Saved {len(self.profiles)} profiles to {save_dir}")
    
    def _save_compressed_profile(self, profile: Dict, path: Path):
        """Save profile in compressed BitDelta format."""
        with open(path, 'wb') as f:
            # Write header
            f.write(b'BITDELTA')  # Magic bytes
            f.write(struct.pack('I', 1))  # Version
            
            # Write metadata
            metadata = json.dumps({
                'model_name': profile['model_name'],
                'compression_ratio': profile['compression_ratio']
            }).encode('utf-8')
            f.write(struct.pack('I', len(metadata)))
            f.write(metadata)
            
            # Write compressed deltas
            for param_name, (signs, scales) in profile['deltas'].items():
                # Write parameter name
                name_bytes = param_name.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                
                # Compress and write delta
                compressed = self.encoder.compress_to_bytes(signs, scales)
                f.write(struct.pack('I', len(compressed)))
                f.write(compressed)
    
    def load_from_disk(self, save_dir: str):
        """Load profiles from disk."""
        save_path = Path(save_dir)
        
        # Load metadata
        with open(save_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.model_name = metadata['model_name']
        self.base_model_path = metadata['base_model_path']
        self.config = BitDeltaConfig(**metadata['config'])
        
        # Load profiles
        for profile_name in metadata['profiles']:
            profile_path = save_path / f'{profile_name}.pt'
            if profile_path.exists():
                self.profiles[profile_name] = torch.load(
                    profile_path, map_location=self.device
                )
                logger.info(f"Loaded profile '{profile_name}'")


class ZenPersonalizationManager:
    """High-level API for Zen model personalization."""
    
    def __init__(self, model_variant: str = 'zen-nano-4b'):
        """
        Initialize personalization manager.
        
        Args:
            model_variant: Zen model variant to use
        """
        self.model_variant = model_variant
        self.bitdelta = ZenBitDelta(model_variant)
        self.active_profile = None
    
    def create_profile_from_examples(self,
                                    profile_name: str,
                                    examples: List[str],
                                    style: Optional[str] = None):
        """
        Create a personalization profile from examples.
        
        Args:
            profile_name: Name for the profile
            examples: Example texts showing desired style
            style: Optional style description
        """
        # Prepare training data
        train_data = []
        for example in examples:
            train_data.append({
                'text': example,
                'style': style or 'personalized'
            })
        
        # Train BitDelta profile
        self.bitdelta.personalize(
            profile_name=profile_name,
            train_data=train_data,
            learning_rate=5e-5,
            epochs=3
        )
        
        logger.info(f"Created profile '{profile_name}' from {len(examples)} examples")
    
    def activate_profile(self, profile_name: str) -> nn.Module:
        """Activate a personalization profile."""
        model = self.bitdelta.load_profile(profile_name)
        self.active_profile = profile_name
        return model
    
    def generate_with_profile(self, 
                             prompt: str,
                             profile_name: Optional[str] = None,
                             max_length: int = 100) -> str:
        """
        Generate text using a specific profile.
        
        Args:
            prompt: Input prompt
            profile_name: Profile to use (or active profile)
            max_length: Maximum generation length
            
        Returns:
            Generated text
        """
        profile = profile_name or self.active_profile
        if not profile:
            raise ValueError("No profile specified or activated")
        
        # Load model with profile
        model = self.activate_profile(profile)
        
        # Tokenize input
        inputs = self.bitdelta.tokenizer(
            prompt, return_tensors='pt'
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.8,
                do_sample=True
            )
        
        # Decode
        generated = self.bitdelta.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        
        return generated
    
    def merge_profiles(self,
                       profiles: List[str],
                       weights: Optional[List[float]] = None,
                       new_name: str = 'merged') -> str:
        """
        Merge multiple profiles into one.
        
        Args:
            profiles: Profile names to merge
            weights: Optional weights for each profile
            new_name: Name for merged profile
            
        Returns:
            Name of merged profile
        """
        if weights is None:
            weights = [1.0 / len(profiles)] * len(profiles)
        
        # Merge BitDelta parameters
        merged_deltas = {}
        
        for param_name in self.bitdelta.profiles[profiles[0]]['deltas'].keys():
            signs_list = []
            scales_list = []
            
            for profile_name in profiles:
                if param_name in self.bitdelta.profiles[profile_name]['deltas']:
                    signs, scales = self.bitdelta.profiles[profile_name]['deltas'][param_name]
                    signs_list.append(signs)
                    scales_list.append(scales)
            
            # Weighted average
            merged_signs = sum(w * s for w, s in zip(weights, signs_list))
            merged_signs = torch.sign(merged_signs)
            
            merged_scales = sum(w * s for w, s in zip(weights, scales_list))
            
            merged_deltas[param_name] = (merged_signs, merged_scales)
        
        # Create merged profile
        self.bitdelta.profiles[new_name] = {
            'model_name': self.model_variant,
            'config': self.bitdelta.config.__dict__,
            'deltas': merged_deltas,
            'compression_ratio': self.bitdelta.profiles[profiles[0]]['compression_ratio']
        }
        
        logger.info(f"Created merged profile '{new_name}' from {profiles}")
        return new_name
    
    def export_profile(self, profile_name: str, output_path: str):
        """Export a profile for sharing."""
        if profile_name not in self.bitdelta.profiles:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in portable format
        profile = self.bitdelta.profiles[profile_name]
        torch.save(profile, output_path)
        
        # Also create human-readable summary
        summary_path = output_path.with_suffix('.json')
        summary = {
            'profile_name': profile_name,
            'model_variant': self.model_variant,
            'compression_ratio': profile['compression_ratio'],
            'num_parameters': sum(s.numel() for s, _ in profile['deltas'].values()),
            'storage_size_mb': sum(
                s.numel() / 8 / 1024 / 1024 + sc.numel() * 4 / 1024 / 1024
                for s, sc in profile['deltas'].values()
            )
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Exported profile '{profile_name}' to {output_path}")


import struct