"""
BitDelta: Efficient 1-bit personalization for large language models.

Store only the sign of weight deltas between base and fine-tuned models,
enabling 10-100x compression of personalized model storage.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import struct
import io


@dataclass
class BitDeltaConfig:
    """Configuration for BitDelta compression."""
    
    # Compression settings
    block_size: int = 128  # Group weights into blocks for better scaling
    use_per_channel_scale: bool = True  # Scale per output channel vs global
    scale_bits: int = 16  # Bits for scale factor storage
    
    # Training settings
    straight_through: bool = True  # Use STE for gradient flow
    delta_regularization: float = 0.01  # L2 regularization on deltas
    sparsity_penalty: float = 0.001  # Encourage sparse deltas
    
    # Optimization
    learning_rate_scale: float = 0.1  # Scale LR for delta-aware training
    warmup_steps: int = 100  # Warmup before applying 1-bit constraint


class BitDeltaEncoder:
    """Encode weight deltas as 1-bit values with scaling factors."""
    
    def __init__(self, config: BitDeltaConfig = None):
        self.config = config or BitDeltaConfig()
    
    def encode(self, base_weights: torch.Tensor, 
               finetuned_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode weight deltas as 1-bit signs + scale factors.
        
        Args:
            base_weights: Original model weights
            finetuned_weights: Fine-tuned model weights
            
        Returns:
            signs: 1-bit sign tensor
            scales: Scale factors for reconstruction
        """
        # Compute deltas
        deltas = finetuned_weights - base_weights
        
        # Extract signs (1-bit)
        signs = torch.sign(deltas)
        
        # Compute scale factors
        if self.config.use_per_channel_scale:
            # Per output channel scaling
            if len(deltas.shape) >= 2:
                scales = torch.abs(deltas).mean(dim=tuple(range(1, len(deltas.shape))))
            else:
                scales = torch.abs(deltas).mean()
        else:
            # Global scaling
            scales = torch.abs(deltas).mean()
        
        return signs, scales
    
    def encode_model(self, base_state: Dict[str, torch.Tensor],
                    finetuned_state: Dict[str, torch.Tensor]) -> Dict[str, Tuple]:
        """Encode entire model state dict."""
        encoded = {}
        
        for name, base_param in base_state.items():
            if name in finetuned_state:
                finetuned_param = finetuned_state[name]
                
                # Only encode parameters that changed
                if not torch.allclose(base_param, finetuned_param, atol=1e-7):
                    signs, scales = self.encode(base_param, finetuned_param)
                    encoded[name] = (signs, scales)
        
        return encoded
    
    def compress_to_bytes(self, signs: torch.Tensor, 
                          scales: torch.Tensor) -> bytes:
        """Pack signs and scales into compact byte representation."""
        buffer = io.BytesIO()
        
        # Write tensor shape
        shape = signs.shape
        buffer.write(struct.pack('I', len(shape)))
        for dim in shape:
            buffer.write(struct.pack('I', dim))
        
        # Pack signs as bits
        flat_signs = signs.flatten()
        num_elements = flat_signs.numel()
        buffer.write(struct.pack('I', num_elements))
        
        # Pack 8 signs per byte
        packed_signs = []
        for i in range(0, num_elements, 8):
            byte = 0
            for j in range(min(8, num_elements - i)):
                if flat_signs[i + j] > 0:
                    byte |= (1 << j)
            packed_signs.append(byte)
        
        buffer.write(bytes(packed_signs))
        
        # Write scales
        if scales.dim() == 0:
            buffer.write(struct.pack('f', scales.item()))
        else:
            buffer.write(struct.pack('I', scales.numel()))
            for scale in scales.flatten():
                buffer.write(struct.pack('f', scale.item()))
        
        return buffer.getvalue()


class BitDeltaDecoder:
    """Decode 1-bit deltas back to full weights."""
    
    def __init__(self, config: BitDeltaConfig = None):
        self.config = config or BitDeltaConfig()
    
    def decode(self, base_weights: torch.Tensor,
               signs: torch.Tensor, 
               scales: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct weights from base + 1-bit deltas.
        
        Args:
            base_weights: Original model weights
            signs: 1-bit sign tensor
            scales: Scale factors
            
        Returns:
            Reconstructed personalized weights
        """
        # Reconstruct deltas
        if self.config.use_per_channel_scale and len(signs.shape) >= 2:
            # Broadcast scales to match tensor shape
            view_shape = [scales.shape[0]] + [1] * (len(signs.shape) - 1)
            scales = scales.view(view_shape)
        
        deltas = signs * scales
        
        # Apply to base weights
        personalized = base_weights + deltas
        
        return personalized
    
    def decode_model(self, base_state: Dict[str, torch.Tensor],
                    encoded_deltas: Dict[str, Tuple]) -> Dict[str, torch.Tensor]:
        """Decode entire model state dict."""
        personalized_state = base_state.copy()
        
        for name, (signs, scales) in encoded_deltas.items():
            if name in base_state:
                personalized_state[name] = self.decode(
                    base_state[name], signs, scales
                )
        
        return personalized_state
    
    def decompress_from_bytes(self, data: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack signs and scales from byte representation."""
        buffer = io.BytesIO(data)
        
        # Read tensor shape
        ndims = struct.unpack('I', buffer.read(4))[0]
        shape = []
        for _ in range(ndims):
            shape.append(struct.unpack('I', buffer.read(4))[0])
        
        # Read packed signs
        num_elements = struct.unpack('I', buffer.read(4))[0]
        num_bytes = (num_elements + 7) // 8
        packed_signs = buffer.read(num_bytes)
        
        # Unpack signs
        flat_signs = []
        for i, byte in enumerate(packed_signs):
            for j in range(8):
                if i * 8 + j < num_elements:
                    sign = 1.0 if (byte & (1 << j)) else -1.0
                    flat_signs.append(sign)
        
        signs = torch.tensor(flat_signs).reshape(shape)
        
        # Read scales
        remaining = buffer.read()
        if len(remaining) == 4:
            scales = torch.tensor(struct.unpack('f', remaining)[0])
        else:
            buffer = io.BytesIO(remaining)
            num_scales = struct.unpack('I', buffer.read(4))[0]
            scales = []
            for _ in range(num_scales):
                scales.append(struct.unpack('f', buffer.read(4))[0])
            scales = torch.tensor(scales)
        
        return signs, scales


class BitDeltaLayer(nn.Module):
    """Layer wrapper with BitDelta support for training."""
    
    def __init__(self, base_layer: nn.Module, config: BitDeltaConfig = None):
        super().__init__()
        self.base_layer = base_layer
        self.config = config or BitDeltaConfig()
        
        # Store base weights (frozen)
        self.register_buffer('base_weight', base_layer.weight.detach().clone())
        if hasattr(base_layer, 'bias') and base_layer.bias is not None:
            self.register_buffer('base_bias', base_layer.bias.detach().clone())
        
        # Learnable delta parameters
        self.delta_signs = nn.Parameter(torch.zeros_like(base_layer.weight))
        self.delta_scales = nn.Parameter(torch.ones(base_layer.weight.shape[0]))
        
        if hasattr(base_layer, 'bias') and base_layer.bias is not None:
            self.bias_delta = nn.Parameter(torch.zeros_like(base_layer.bias))
    
    def get_quantized_weight(self) -> torch.Tensor:
        """Get weight with 1-bit quantized deltas."""
        if self.training and self.config.straight_through:
            # Straight-through estimator for training
            signs = torch.sign(self.delta_signs)
            quantized = self.delta_signs + (signs - self.delta_signs).detach()
        else:
            quantized = torch.sign(self.delta_signs)
        
        # Apply scales
        if self.config.use_per_channel_scale:
            scales = self.delta_scales.view(-1, *([1] * (len(self.delta_signs.shape) - 1)))
            deltas = quantized * scales
        else:
            deltas = quantized * self.delta_scales.mean()
        
        return self.base_weight + deltas
    
    def forward(self, x):
        """Forward pass with BitDelta weights."""
        weight = self.get_quantized_weight()
        
        if hasattr(self, 'bias_delta'):
            bias = self.base_bias + self.bias_delta
        elif hasattr(self.base_layer, 'bias'):
            bias = self.base_layer.bias
        else:
            bias = None
        
        # Call appropriate forward function based on layer type
        if isinstance(self.base_layer, nn.Linear):
            return nn.functional.linear(x, weight, bias)
        elif isinstance(self.base_layer, nn.Conv2d):
            return nn.functional.conv2d(
                x, weight, bias,
                self.base_layer.stride,
                self.base_layer.padding,
                self.base_layer.dilation,
                self.base_layer.groups
            )
        else:
            raise NotImplementedError(f"BitDelta not implemented for {type(self.base_layer)}")
    
    def regularization_loss(self) -> torch.Tensor:
        """Compute regularization terms for BitDelta training."""
        loss = 0.0
        
        # L2 regularization on delta magnitudes
        if self.config.delta_regularization > 0:
            loss += self.config.delta_regularization * (
                self.delta_signs.pow(2).mean() + 
                self.delta_scales.pow(2).mean()
            )
        
        # Sparsity penalty
        if self.config.sparsity_penalty > 0:
            loss += self.config.sparsity_penalty * torch.abs(self.delta_signs).mean()
        
        return loss


class BitDeltaModel(nn.Module):
    """Wrapper for models with BitDelta personalization."""
    
    def __init__(self, base_model: nn.Module, config: BitDeltaConfig = None):
        super().__init__()
        self.config = config or BitDeltaConfig()
        self.base_model = base_model
        
        # Replace layers with BitDelta versions
        self._replace_layers()
    
    def _replace_layers(self):
        """Replace standard layers with BitDelta layers."""
        for name, module in self.base_model.named_children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                setattr(self.base_model, name, 
                       BitDeltaLayer(module, self.config))
            elif len(list(module.children())) > 0:
                # Recursive replacement
                self._replace_module_layers(module)
    
    def _replace_module_layers(self, module: nn.Module):
        """Recursively replace layers in a module."""
        for name, child in module.named_children():
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                setattr(module, name, 
                       BitDeltaLayer(child, self.config))
            elif len(list(child.children())) > 0:
                self._replace_module_layers(child)
    
    def forward(self, *args, **kwargs):
        """Forward pass through base model with BitDelta weights."""
        return self.base_model(*args, **kwargs)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Get total regularization loss from all BitDelta layers."""
        total_loss = 0.0
        for module in self.modules():
            if isinstance(module, BitDeltaLayer):
                total_loss += module.regularization_loss()
        return total_loss
    
    def export_bitdelta(self) -> Dict[str, Tuple]:
        """Export current BitDelta parameters."""
        deltas = {}
        for name, module in self.named_modules():
            if isinstance(module, BitDeltaLayer):
                signs = torch.sign(module.delta_signs)
                scales = module.delta_scales
                deltas[f"{name}.weight"] = (signs, scales)
                
                if hasattr(module, 'bias_delta'):
                    deltas[f"{name}.bias"] = (
                        torch.sign(module.bias_delta),
                        torch.abs(module.bias_delta)
                    )
        return deltas


def compute_compression_ratio(base_model: nn.Module, 
                             bitdelta_params: Dict) -> float:
    """Compute storage compression ratio."""
    # Original model size
    original_size = 0
    for param in base_model.parameters():
        original_size += param.numel() * 4  # 32-bit floats
    
    # BitDelta size
    bitdelta_size = 0
    for name, (signs, scales) in bitdelta_params.items():
        bitdelta_size += signs.numel() / 8  # 1 bit per parameter
        bitdelta_size += scales.numel() * 4  # 32-bit scales
    
    return original_size / bitdelta_size if bitdelta_size > 0 else float('inf')