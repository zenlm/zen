"""
Test suite for BitDelta implementation.

Tests core functionality, compression ratios, and quality retention.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import pytest
import tempfile
from pathlib import Path

from bitdelta import (
    BitDeltaConfig, BitDeltaEncoder, BitDeltaDecoder,
    BitDeltaLayer, BitDeltaModel, compute_compression_ratio
)
from training import (
    BitDeltaTrainer, ProgressiveQuantizationTrainer, 
    MultiProfileTrainer
)


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=768):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def test_bitdelta_encoding():
    """Test basic encoding and decoding."""
    config = BitDeltaConfig()
    encoder = BitDeltaEncoder(config)
    decoder = BitDeltaDecoder(config)
    
    # Create test tensors
    base_weights = torch.randn(256, 128)
    delta = torch.randn(256, 128) * 0.1  # Small delta
    finetuned_weights = base_weights + delta
    
    # Encode
    signs, scales = encoder.encode(base_weights, finetuned_weights)
    
    # Check shapes
    assert signs.shape == base_weights.shape
    assert scales.shape == (256,) or scales.dim() == 0
    
    # Check signs are binary
    assert torch.all((signs == 1) | (signs == -1) | (signs == 0))
    
    # Decode
    reconstructed = decoder.decode(base_weights, signs, scales)
    
    # Check reconstruction error
    error = torch.mean(torch.abs(reconstructed - finetuned_weights))
    relative_error = error / torch.mean(torch.abs(finetuned_weights))
    
    print(f"Reconstruction error: {error:.6f}")
    print(f"Relative error: {relative_error:.4%}")
    
    # Should have reasonable reconstruction
    assert relative_error < 0.5  # Less than 50% error


def test_compression_ratio():
    """Test compression ratio calculation."""
    model = SimpleModel()
    config = BitDeltaConfig()
    
    # Create BitDelta model
    bitdelta_model = BitDeltaModel(model, config)
    
    # Simulate training (just set some delta values)
    for module in bitdelta_model.modules():
        if isinstance(module, BitDeltaLayer):
            module.delta_signs.data = torch.sign(torch.randn_like(module.delta_signs))
            module.delta_scales.data = torch.abs(torch.randn_like(module.delta_scales))
    
    # Export BitDelta parameters
    bitdelta_params = bitdelta_model.export_bitdelta()
    
    # Calculate compression ratio
    ratio = compute_compression_ratio(model, bitdelta_params)
    
    print(f"Compression ratio: {ratio:.1f}x")
    
    # Should achieve significant compression
    assert ratio > 30  # At least 30x compression


def test_bitdelta_layer():
    """Test BitDelta layer functionality."""
    config = BitDeltaConfig()
    
    # Create base layer
    base_layer = nn.Linear(128, 64)
    base_weight = base_layer.weight.clone()
    
    # Create BitDelta layer
    bitdelta_layer = BitDeltaLayer(base_layer, config)
    
    # Test forward pass
    x = torch.randn(32, 128)
    output = bitdelta_layer(x)
    
    assert output.shape == (32, 64)
    
    # Initially should match base layer
    base_output = nn.functional.linear(x, base_weight, base_layer.bias)
    initial_diff = torch.mean(torch.abs(output - base_output))
    assert initial_diff < 1e-5
    
    # Modify deltas
    bitdelta_layer.delta_signs.data = torch.sign(torch.randn_like(bitdelta_layer.delta_signs))
    bitdelta_layer.delta_scales.data = torch.ones_like(bitdelta_layer.delta_scales) * 0.1
    
    # Should now differ from base
    output2 = bitdelta_layer(x)
    diff = torch.mean(torch.abs(output2 - base_output))
    assert diff > 0.01
    
    # Test regularization loss
    reg_loss = bitdelta_layer.regularization_loss()
    assert reg_loss > 0


def test_byte_compression():
    """Test byte-level compression and decompression."""
    config = BitDeltaConfig()
    encoder = BitDeltaEncoder(config)
    decoder = BitDeltaDecoder(config)
    
    # Create test data
    signs = torch.sign(torch.randn(64, 32))
    scales = torch.abs(torch.randn(64))
    
    # Compress to bytes
    compressed = encoder.compress_to_bytes(signs, scales)
    
    # Check compression
    original_size = signs.numel() * 4 + scales.numel() * 4  # 32-bit floats
    compressed_size = len(compressed)
    
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression: {original_size / compressed_size:.1f}x")
    
    # Decompress
    signs_dec, scales_dec = decoder.decompress_from_bytes(compressed)
    
    # Check correctness
    assert torch.allclose(signs, signs_dec)
    assert torch.allclose(scales, scales_dec)


def test_progressive_quantization():
    """Test progressive quantization training."""
    model = SimpleModel()
    config = BitDeltaConfig()
    
    # Create trainer with custom schedule
    schedule = [
        (0, 0.0),
        (10, 0.5),
        (20, 1.0)
    ]
    
    trainer = ProgressiveQuantizationTrainer(
        model, config, device='cpu', 
        quantization_schedule=schedule
    )
    
    # Check quantization levels at different steps
    trainer.step = 0
    assert trainer.get_quantization_level() == 0.0
    
    trainer.step = 10
    assert trainer.get_quantization_level() == 0.5
    
    trainer.step = 25
    assert trainer.get_quantization_level() == 1.0


def test_multi_profile():
    """Test multi-profile management."""
    model = SimpleModel()
    config = BitDeltaConfig()
    
    manager = MultiProfileTrainer(model, config, device='cpu')
    
    # Create profiles
    manager.create_profile('profile1')
    manager.create_profile('profile2')
    
    assert 'profile1' in manager.profiles
    assert 'profile2' in manager.profiles
    
    # Simulate adding deltas to profiles
    fake_deltas = {
        'layer1.weight': (
            torch.sign(torch.randn(2048, 768)),
            torch.ones(2048)
        )
    }
    
    manager.profiles['profile1'] = fake_deltas
    manager.profiles['profile2'] = fake_deltas
    
    # Test profile merging
    merged = manager.merge_profiles(
        ['profile1', 'profile2'],
        weights=[0.7, 0.3],
        new_profile_name='merged'
    )
    
    assert 'merged' in manager.profiles
    
    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        manager.save_profiles(tmpdir)
        
        # Create new manager and load
        manager2 = MultiProfileTrainer(model, config, device='cpu')
        manager2.load_profiles(tmpdir)
        
        assert 'profile1' in manager2.profiles
        assert 'profile2' in manager2.profiles


def test_model_state_encoding():
    """Test encoding/decoding of full model state dict."""
    config = BitDeltaConfig()
    encoder = BitDeltaEncoder(config)
    decoder = BitDeltaDecoder(config)
    
    # Create two models
    base_model = SimpleModel()
    finetuned_model = SimpleModel()
    
    # Modify finetuned model
    with torch.no_grad():
        for param in finetuned_model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    # Encode state dict differences
    base_state = base_model.state_dict()
    finetuned_state = finetuned_model.state_dict()
    
    encoded = encoder.encode_model(base_state, finetuned_state)
    
    # Should have encoded the linear layers
    assert len(encoded) > 0
    
    # Decode
    reconstructed_state = decoder.decode_model(base_state, encoded)
    
    # Check reconstruction
    for name, param in reconstructed_state.items():
        if name in finetuned_state:
            original = finetuned_state[name]
            error = torch.mean(torch.abs(param - original))
            relative_error = error / (torch.mean(torch.abs(original)) + 1e-8)
            
            print(f"{name}: error={error:.6f}, relative={relative_error:.4%}")
            
            # Should have reasonable reconstruction
            assert relative_error < 1.0  # Less than 100% error


def test_straight_through_estimator():
    """Test STE gradient flow in training mode."""
    config = BitDeltaConfig(straight_through=True)
    
    base_layer = nn.Linear(10, 5)
    bitdelta_layer = BitDeltaLayer(base_layer, config)
    
    # Set to training mode
    bitdelta_layer.train()
    
    # Forward pass
    x = torch.randn(2, 10, requires_grad=True)
    output = bitdelta_layer(x)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients flow through
    assert bitdelta_layer.delta_signs.grad is not None
    assert bitdelta_layer.delta_scales.grad is not None
    assert x.grad is not None


def test_selective_layer_replacement():
    """Test selective replacement of specific layers."""
    model = SimpleModel()
    config = BitDeltaConfig()
    
    bitdelta_model = BitDeltaModel(model, config)
    
    # Count BitDelta layers
    bitdelta_count = sum(
        1 for m in bitdelta_model.modules() 
        if isinstance(m, BitDeltaLayer)
    )
    
    # Should have replaced the linear layers
    assert bitdelta_count == 2  # layer1 and layer2


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("BitDelta Test Suite")
    print("=" * 60)
    
    tests = [
        ("Encoding/Decoding", test_bitdelta_encoding),
        ("Compression Ratio", test_compression_ratio),
        ("BitDelta Layer", test_bitdelta_layer),
        ("Byte Compression", test_byte_compression),
        ("Progressive Quantization", test_progressive_quantization),
        ("Multi-Profile", test_multi_profile),
        ("Model State Encoding", test_model_state_encoding),
        ("Straight-Through Estimator", test_straight_through_estimator),
        ("Selective Layer Replacement", test_selective_layer_replacement)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 40)
        
        try:
            test_func()
            print(f"✓ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)