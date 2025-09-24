"""
BitDelta integration for Zen model personalization
Efficient parameter-efficient fine-tuning with binary deltas
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json

# Import BitDelta from the existing implementation
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "bitdelta"))
try:
    from bitdelta import BitDelta
    from training import BitDeltaTrainer
except ImportError:
    print("Warning: BitDelta not found. Installing mock implementation.")
    BitDelta = None
    BitDeltaTrainer = None


class ZenBitDelta:
    """BitDelta integration for Zen models"""
    
    def __init__(
        self,
        model: nn.Module,
        compression_ratio: float = 0.1,
        block_size: int = 64,
        quantization_bits: int = 1,
        learning_rate: float = 1e-3,
        use_residual: bool = True
    ):
        self.model = model
        self.compression_ratio = compression_ratio
        self.block_size = block_size
        self.quantization_bits = quantization_bits
        self.learning_rate = learning_rate
        self.use_residual = use_residual
        
        self.bitdelta_layers = {}
        self.original_params = {}
        
    def apply_bitdelta(self, target_modules: list = None):
        """Apply BitDelta to specified modules"""
        if target_modules is None:
            target_modules = self._get_default_target_modules()
        
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    self._apply_bitdelta_to_linear(name, module)
                elif isinstance(module, nn.Conv2d):
                    self._apply_bitdelta_to_conv(name, module)
        
        print(f"Applied BitDelta to {len(self.bitdelta_layers)} layers")
        
    def _apply_bitdelta_to_linear(self, name: str, module: nn.Linear):
        """Apply BitDelta to Linear layer"""
        weight = module.weight.data
        
        # Store original parameters
        self.original_params[name] = {
            'weight': weight.clone(),
            'bias': module.bias.data.clone() if module.bias is not None else None
        }
        
        # Create BitDelta representation
        if BitDelta is not None:
            bitdelta = BitDelta(
                weight.shape,
                compression_ratio=self.compression_ratio,
                block_size=self.block_size,
                quantization_bits=self.quantization_bits
            )
            
            # Compress weights
            compressed = bitdelta.compress(weight)
            self.bitdelta_layers[name] = {
                'type': 'linear',
                'bitdelta': bitdelta,
                'compressed': compressed,
                'shape': weight.shape
            }
        else:
            # Fallback implementation
            self.bitdelta_layers[name] = {
                'type': 'linear',
                'delta': torch.zeros_like(weight),
                'mask': self._create_sparse_mask(weight.shape),
                'shape': weight.shape
            }
    
    def _apply_bitdelta_to_conv(self, name: str, module: nn.Conv2d):
        """Apply BitDelta to Conv2d layer"""
        weight = module.weight.data
        
        # Store original parameters
        self.original_params[name] = {
            'weight': weight.clone(),
            'bias': module.bias.data.clone() if module.bias is not None else None
        }
        
        # Create BitDelta representation
        if BitDelta is not None:
            bitdelta = BitDelta(
                weight.shape,
                compression_ratio=self.compression_ratio,
                block_size=self.block_size,
                quantization_bits=self.quantization_bits
            )
            
            compressed = bitdelta.compress(weight)
            self.bitdelta_layers[name] = {
                'type': 'conv',
                'bitdelta': bitdelta,
                'compressed': compressed,
                'shape': weight.shape
            }
        else:
            self.bitdelta_layers[name] = {
                'type': 'conv',
                'delta': torch.zeros_like(weight),
                'mask': self._create_sparse_mask(weight.shape),
                'shape': weight.shape
            }
    
    def _create_sparse_mask(self, shape: torch.Size) -> torch.Tensor:
        """Create sparse mask for weight updates"""
        mask = torch.zeros(shape)
        num_params = np.prod(shape)
        num_active = int(num_params * self.compression_ratio)
        
        # Randomly select parameters to update
        indices = torch.randperm(num_params)[:num_active]
        mask.view(-1)[indices] = 1.0
        
        return mask
    
    def _get_default_target_modules(self) -> list:
        """Get default target modules for BitDelta"""
        return [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
            'fc1', 'fc2', 'mlp'
        ]
    
    def forward_with_delta(self, *args, **kwargs):
        """Forward pass with BitDelta modifications"""
        # Apply deltas to weights
        with torch.no_grad():
            for name, layer_info in self.bitdelta_layers.items():
                module = self._get_module_by_name(name)
                
                if BitDelta is not None and 'bitdelta' in layer_info:
                    # Decompress and apply
                    delta = layer_info['bitdelta'].decompress(layer_info['compressed'])
                    module.weight.data = self.original_params[name]['weight'] + delta
                else:
                    # Simple delta application
                    delta = layer_info['delta'] * layer_info['mask']
                    module.weight.data = self.original_params[name]['weight'] + delta
        
        # Forward pass
        output = self.model(*args, **kwargs)
        
        return output
    
    def _get_module_by_name(self, name: str) -> nn.Module:
        """Get module by name from model"""
        parts = name.split('.')
        module = self.model
        for part in parts:
            module = getattr(module, part)
        return module
    
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Single training step with BitDelta"""
        optimizer.zero_grad()
        
        # Forward pass with deltas
        outputs = self.forward_with_delta(inputs)
        
        # Compute loss
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Update only delta parameters
        with torch.no_grad():
            for name, layer_info in self.bitdelta_layers.items():
                module = self._get_module_by_name(name)
                
                if module.weight.grad is not None:
                    if BitDelta is not None and 'bitdelta' in layer_info:
                        # Update compressed representation
                        grad = module.weight.grad * self.learning_rate
                        layer_info['compressed'] = layer_info['bitdelta'].update(
                            layer_info['compressed'], grad
                        )
                    else:
                        # Update delta with mask
                        layer_info['delta'] -= self.learning_rate * module.weight.grad * layer_info['mask']
        
        return {'loss': loss.item()}
    
    def save_deltas(self, path: str):
        """Save BitDelta parameters"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'compression_ratio': self.compression_ratio,
            'block_size': self.block_size,
            'quantization_bits': self.quantization_bits,
            'layers': {}
        }
        
        for name, layer_info in self.bitdelta_layers.items():
            if BitDelta is not None and 'bitdelta' in layer_info:
                save_dict['layers'][name] = {
                    'type': layer_info['type'],
                    'compressed': layer_info['compressed'].cpu().numpy().tolist(),
                    'shape': list(layer_info['shape'])
                }
            else:
                save_dict['layers'][name] = {
                    'type': layer_info['type'],
                    'delta': layer_info['delta'].cpu().numpy().tolist(),
                    'mask': layer_info['mask'].cpu().numpy().tolist(),
                    'shape': list(layer_info['shape'])
                }
        
        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"BitDelta parameters saved to {save_path}")
    
    def load_deltas(self, path: str):
        """Load BitDelta parameters"""
        with open(path, 'r') as f:
            save_dict = json.load(f)
        
        self.compression_ratio = save_dict['compression_ratio']
        self.block_size = save_dict['block_size']
        self.quantization_bits = save_dict['quantization_bits']
        
        self.bitdelta_layers = {}
        for name, layer_data in save_dict['layers'].items():
            if 'compressed' in layer_data:
                self.bitdelta_layers[name] = {
                    'type': layer_data['type'],
                    'compressed': torch.tensor(layer_data['compressed']),
                    'shape': torch.Size(layer_data['shape'])
                }
            else:
                self.bitdelta_layers[name] = {
                    'type': layer_data['type'],
                    'delta': torch.tensor(layer_data['delta']),
                    'mask': torch.tensor(layer_data['mask']),
                    'shape': torch.Size(layer_data['shape'])
                }
        
        print(f"BitDelta parameters loaded from {path}")
    
    def merge_deltas(self, output_path: str = None):
        """Merge BitDelta with original model"""
        with torch.no_grad():
            for name, layer_info in self.bitdelta_layers.items():
                module = self._get_module_by_name(name)
                
                if BitDelta is not None and 'bitdelta' in layer_info:
                    delta = layer_info['bitdelta'].decompress(layer_info['compressed'])
                else:
                    delta = layer_info['delta'] * layer_info['mask']
                
                # Permanently apply delta
                module.weight.data = self.original_params[name]['weight'] + delta
        
        if output_path:
            # Save merged model
            torch.save(self.model.state_dict(), output_path)
            print(f"Merged model saved to {output_path}")
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        total_params = 0
        compressed_params = 0
        
        for name, layer_info in self.bitdelta_layers.items():
            shape = layer_info['shape']
            total = np.prod(shape)
            total_params += total
            
            if BitDelta is not None and 'bitdelta' in layer_info:
                # Actual BitDelta compression
                compressed = layer_info['compressed'].numel()
            else:
                # Sparse mask compression
                compressed = layer_info['mask'].sum().item()
            
            compressed_params += compressed
        
        return {
            'total_parameters': total_params,
            'compressed_parameters': compressed_params,
            'compression_ratio': compressed_params / total_params if total_params > 0 else 0,
            'memory_saved_mb': (total_params - compressed_params) * 4 / (1024 * 1024),
            'num_layers': len(self.bitdelta_layers)
        }


def create_bitdelta_trainer(
    model: nn.Module,
    config: Dict[str, Any],
    train_dataset,
    val_dataset
) -> ZenBitDelta:
    """Create BitDelta trainer for Zen models"""
    
    bitdelta = ZenBitDelta(
        model=model,
        compression_ratio=config.get('compression_ratio', 0.1),
        block_size=config.get('block_size', 64),
        quantization_bits=config.get('quantization_bits', 1),
        learning_rate=config.get('learning_rate', 1e-3),
        use_residual=config.get('use_residual', True)
    )
    
    # Apply BitDelta to model
    target_modules = config.get('target_modules', None)
    bitdelta.apply_bitdelta(target_modules)
    
    # Print compression stats
    stats = bitdelta.get_compression_stats()
    print(f"BitDelta Compression Stats:")
    print(f"  Total Parameters: {stats['total_parameters']:,}")
    print(f"  Compressed Parameters: {stats['compressed_parameters']:,}")
    print(f"  Compression Ratio: {stats['compression_ratio']:.2%}")
    print(f"  Memory Saved: {stats['memory_saved_mb']:.2f} MB")
    
    return bitdelta