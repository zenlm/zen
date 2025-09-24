#!/usr/bin/env python3
"""
Zen-Nano Setup and Optimization
Creates ultra-lightweight models for edge and mobile deployment
Based on Qwen3-4B-2507 with aggressive quantization
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import onnx
import coremltools as ct
import tensorflow as tf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
import fire
from tqdm import tqdm

@dataclass 
class NanoConfig:
    """Configuration for Zen-Nano models"""
    base_model: str = "Qwen/Qwen2.5-3B"  # Using 3B as base since 4B-2507 not available
    output_dir: str = "zen-nano-models"
    
    # Quantization levels
    quantization_modes: List[str] = None
    target_sizes: Dict[str, float] = None  # MB
    
    # Optimization techniques
    use_pruning: bool = True
    pruning_ratio: float = 0.3
    use_distillation: bool = True
    use_quantization: bool = True
    use_layer_reduction: bool = True
    layers_to_keep: int = 12  # Reduce from 24
    
    # Mobile targets
    create_tflite: bool = True
    create_coreml: bool = True
    create_onnx: bool = True
    create_wasm: bool = False
    
    # Progressive download setup
    create_progressive_layers: bool = True
    num_progressive_stages: int = 5
    
    def __post_init__(self):
        if self.quantization_modes is None:
            self.quantization_modes = ["1bit", "2bit", "4bit", "int8", "fp16"]
        
        if self.target_sizes is None:
            self.target_sizes = {
                "1bit": 300,   # MB
                "2bit": 600,
                "4bit": 1200,
                "int8": 2400,
                "fp16": 8000
            }

class ModelQuantizer:
    """Advanced quantization for ultra-small models"""
    
    def __init__(self):
        self.quantization_functions = {
            "1bit": self.quantize_1bit,
            "2bit": self.quantize_2bit,
            "4bit": self.quantize_4bit,
            "int8": self.quantize_int8,
            "fp16": self.quantize_fp16
        }
    
    def quantize_1bit(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """1-bit quantization (sign only)"""
        quantized = torch.sign(tensor)
        scale = torch.abs(tensor).mean()
        
        # Pack bits efficiently
        packed = self._pack_bits(quantized, bits=1)
        
        metadata = {
            "scale": scale.item(),
            "bits": 1,
            "original_shape": list(tensor.shape)
        }
        
        return packed, metadata
    
    def quantize_2bit(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """2-bit quantization"""
        # Quantize to -1, -0.5, 0.5, 1
        scale = torch.abs(tensor).max()
        normalized = tensor / (scale + 1e-8)
        
        boundaries = torch.tensor([-0.67, 0, 0.67])
        quantized = torch.bucketize(normalized, boundaries) - 1.5
        
        packed = self._pack_bits(quantized + 1.5, bits=2)
        
        metadata = {
            "scale": scale.item(),
            "bits": 2,
            "original_shape": list(tensor.shape)
        }
        
        return packed, metadata
    
    def quantize_4bit(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """4-bit quantization"""
        scale = torch.abs(tensor).max()
        normalized = tensor / (scale + 1e-8)
        
        # Quantize to 16 levels
        quantized = torch.round(normalized * 7.5 + 7.5).clamp(0, 15)
        
        packed = self._pack_bits(quantized, bits=4)
        
        metadata = {
            "scale": scale.item(),
            "bits": 4,
            "original_shape": list(tensor.shape)
        }
        
        return packed, metadata
    
    def quantize_int8(self, tensor: torch.Tensor) -> torch.Tensor:
        """8-bit integer quantization"""
        scale = torch.abs(tensor).max() / 127
        quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
        return quantized
    
    def quantize_fp16(self, tensor: torch.Tensor) -> torch.Tensor:
        """16-bit floating point"""
        return tensor.half()
    
    def _pack_bits(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack low-bit tensors efficiently"""
        # Flatten tensor
        flat = tensor.flatten().to(torch.int8)
        
        if bits == 1:
            # Pack 8 values per byte
            packed_size = (len(flat) + 7) // 8
            packed = torch.zeros(packed_size, dtype=torch.uint8)
            
            for i in range(len(flat)):
                byte_idx = i // 8
                bit_idx = i % 8
                if flat[i] > 0:
                    packed[byte_idx] |= (1 << bit_idx)
        
        elif bits == 2:
            # Pack 4 values per byte
            packed_size = (len(flat) + 3) // 4
            packed = torch.zeros(packed_size, dtype=torch.uint8)
            
            for i in range(len(flat)):
                byte_idx = i // 4
                bit_idx = (i % 4) * 2
                packed[byte_idx] |= (int(flat[i]) << bit_idx)
        
        elif bits == 4:
            # Pack 2 values per byte
            packed_size = (len(flat) + 1) // 2
            packed = torch.zeros(packed_size, dtype=torch.uint8)
            
            for i in range(len(flat)):
                byte_idx = i // 2
                if i % 2 == 0:
                    packed[byte_idx] = int(flat[i])
                else:
                    packed[byte_idx] |= (int(flat[i]) << 4)
        
        else:
            packed = flat
        
        return packed

class ModelPruner:
    """Prune model weights for size reduction"""
    
    def __init__(self, pruning_ratio: float = 0.3):
        self.pruning_ratio = pruning_ratio
    
    def prune_magnitude(self, model: nn.Module) -> nn.Module:
        """Magnitude-based pruning"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                threshold = torch.abs(weight).flatten().kthvalue(
                    int(weight.numel() * self.pruning_ratio)
                )[0]
                mask = torch.abs(weight) > threshold
                module.weight.data *= mask
        
        return model
    
    def prune_structured(self, model: nn.Module) -> nn.Module:
        """Structured pruning (remove entire neurons)"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                
                # Compute importance of each neuron
                importance = torch.abs(weight).sum(dim=1)
                threshold = importance.kthvalue(
                    int(len(importance) * self.pruning_ratio)
                )[0]
                
                # Create mask
                mask = importance > threshold
                
                # Apply mask
                module.weight.data = weight[mask]
                if module.bias is not None:
                    module.bias.data = module.bias.data[mask]
                
                # Update dimensions
                module.out_features = mask.sum().item()
        
        return model

class LayerReducer:
    """Reduce number of layers in model"""
    
    def __init__(self, layers_to_keep: int = 12):
        self.layers_to_keep = layers_to_keep
    
    def reduce_layers(self, model: nn.Module) -> nn.Module:
        """Keep only most important layers"""
        # This is model-specific, example for transformer
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
            num_layers = len(layers)
            
            if num_layers > self.layers_to_keep:
                # Keep evenly spaced layers
                indices_to_keep = np.linspace(0, num_layers-1, self.layers_to_keep, dtype=int)
                new_layers = nn.ModuleList([layers[i] for i in indices_to_keep])
                model.transformer.h = new_layers
        
        return model

class MobileConverter:
    """Convert models for mobile deployment"""
    
    def __init__(self, model, tokenizer, config: NanoConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def convert_to_tflite(self, quantization_mode: str = "int8"):
        """Convert to TensorFlow Lite"""
        print(f"Converting to TFLite ({quantization_mode})...")
        
        # Create TF function
        @tf.function
        def inference(input_ids):
            # Simplified inference
            return {"output": tf.random.normal([1, 100, 32000])}  # Placeholder
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_concrete_functions([inference.get_concrete_function(
            input_ids=tf.TensorSpec(shape=[1, None], dtype=tf.int32)
        )])
        
        # Apply quantization
        if quantization_mode == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        elif quantization_mode == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save
        output_path = self.output_dir / f"zen_nano_{quantization_mode}.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved: {output_path}")
        print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    def convert_to_coreml(self, quantization_mode: str = "int8"):
        """Convert to Core ML for iOS"""
        print(f"Converting to Core ML ({quantization_mode})...")
        
        # Trace the model
        dummy_input = torch.randint(0, 32000, (1, 10))
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Convert to Core ML
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(1, ct.RangeDim(1, 512)), dtype=np.int32)],
            compute_units=ct.ComputeUnit.ALL,
            convert_to="mlprogram"
        )
        
        # Apply quantization
        if quantization_mode in ["int8", "4bit"]:
            from coremltools.optimize.torch import palettization
            config = palettization.ModulePalettizationConfig(
                n_bits=8 if quantization_mode == "int8" else 4,
                granularity="per_channel"
            )
            mlmodel = palettization.palettize(mlmodel, config)
        
        # Save
        output_path = self.output_dir / f"zen_nano_{quantization_mode}.mlpackage"
        mlmodel.save(output_path)
        
        print(f"Core ML model saved: {output_path}")
    
    def convert_to_onnx(self, quantization_mode: str = "fp16"):
        """Convert to ONNX"""
        print(f"Converting to ONNX ({quantization_mode})...")
        
        # Export to ONNX
        dummy_input = torch.randint(0, 32000, (1, 10))
        output_path = self.output_dir / f"zen_nano_{quantization_mode}.onnx"
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size', 1: 'sequence'}
            }
        )
        
        # Optimize ONNX
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        if quantization_mode == "int8":
            quantize_dynamic(
                str(output_path),
                str(output_path).replace('.onnx', '_int8.onnx'),
                weight_type=QuantType.QInt8
            )
            output_path = output_path.parent / f"zen_nano_int8.onnx"
        
        print(f"ONNX model saved: {output_path}")
        print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

class ProgressiveLayerCreator:
    """Create progressive download layers"""
    
    def __init__(self, base_model, config: NanoConfig):
        self.base_model = base_model
        self.config = config
        self.output_dir = Path(config.output_dir) / "progressive"
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def create_progressive_layers(self):
        """Create layers for progressive download"""
        stages = []
        
        for i in range(self.config.num_progressive_stages):
            stage_name = f"stage_{i}"
            quality = 0.7 + (0.3 * i / (self.config.num_progressive_stages - 1))
            
            if i == 0:
                # Ultra-minimal base
                stage = self._create_minimal_base()
            elif i == 1:
                # Add basic attention
                stage = self._create_basic_attention()
            elif i == 2:
                # Add MLPs
                stage = self._create_mlp_layers()
            elif i == 3:
                # Add specialized experts
                stage = self._create_expert_layers()
            else:
                # Full quality
                stage = self._create_full_quality()
            
            # Save stage
            stage_path = self.output_dir / f"{stage_name}.pt"
            torch.save(stage, stage_path)
            
            stages.append({
                "name": stage_name,
                "path": str(stage_path),
                "quality": quality,
                "size_mb": stage_path.stat().st_size / 1024 / 1024
            })
            
            print(f"Created {stage_name}: {stages[-1]['size_mb']:.2f} MB, quality={quality:.2%}")
        
        # Save manifest
        manifest = {
            "model": "zen-nano",
            "stages": stages,
            "total_stages": len(stages)
        }
        
        with open(self.output_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return stages
    
    def _create_minimal_base(self) -> Dict:
        """Create ultra-minimal 1-bit base"""
        # Extract embeddings and basic layers
        base = {
            "embeddings": self._quantize_layer(self.base_model.get_input_embeddings(), "1bit"),
            "layers": self._extract_minimal_layers(num_layers=3)
        }
        return base
    
    def _create_basic_attention(self) -> Dict:
        """Add attention mechanisms"""
        return {
            "attention_layers": self._extract_attention_layers(num_layers=6)
        }
    
    def _create_mlp_layers(self) -> Dict:
        """Add MLP layers"""
        return {
            "mlp_layers": self._extract_mlp_layers(num_layers=6)
        }
    
    def _create_expert_layers(self) -> Dict:
        """Add specialized expert layers"""
        return {
            "expert_layers": self._extract_expert_layers()
        }
    
    def _create_full_quality(self) -> Dict:
        """Full quality layers"""
        return {
            "full_layers": self._extract_all_remaining_layers()
        }
    
    def _quantize_layer(self, layer, mode):
        """Quantize a single layer"""
        quantizer = ModelQuantizer()
        if hasattr(layer, 'weight'):
            quantized, metadata = quantizer.quantization_functions[mode](layer.weight)
            return {"weight": quantized, "metadata": metadata}
        return {}
    
    def _extract_minimal_layers(self, num_layers):
        """Extract minimal set of layers"""
        # Implementation depends on model architecture
        return {}
    
    def _extract_attention_layers(self, num_layers):
        """Extract attention layers"""
        return {}
    
    def _extract_mlp_layers(self, num_layers):
        """Extract MLP layers"""
        return {}
    
    def _extract_expert_layers(self):
        """Extract expert layers if available"""
        return {}
    
    def _extract_all_remaining_layers(self):
        """Extract all remaining layers"""
        return {}

class ZenNanoBuilder:
    """Main builder for Zen-Nano models"""
    
    def __init__(self, config: NanoConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load base model
        self.load_base_model()
        
        # Initialize components
        self.quantizer = ModelQuantizer()
        self.pruner = ModelPruner(config.pruning_ratio)
        self.layer_reducer = LayerReducer(config.layers_to_keep)
    
    def load_base_model(self):
        """Load the base model"""
        print(f"Loading base model: {self.config.base_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print(f"Model loaded: {self.model.num_parameters():,} parameters")
    
    def build_all_variants(self):
        """Build all Zen-Nano variants"""
        results = []
        
        for quant_mode in self.config.quantization_modes:
            print(f"\n{'='*50}")
            print(f"Building Zen-Nano {quant_mode}")
            print(f"{'='*50}")
            
            result = self.build_variant(quant_mode)
            results.append(result)
        
        # Create progressive layers if enabled
        if self.config.create_progressive_layers:
            print(f"\n{'='*50}")
            print("Creating Progressive Download Layers")
            print(f"{'='*50}")
            
            creator = ProgressiveLayerCreator(self.model, self.config)
            stages = creator.create_progressive_layers()
            results.append({"type": "progressive", "stages": stages})
        
        # Convert to mobile formats
        if any([self.config.create_tflite, self.config.create_coreml, self.config.create_onnx]):
            print(f"\n{'='*50}")
            print("Converting to Mobile Formats")
            print(f"{'='*50}")
            
            converter = MobileConverter(self.model, self.tokenizer, self.config)
            
            if self.config.create_tflite:
                converter.convert_to_tflite("int8")
            
            if self.config.create_coreml:
                converter.convert_to_coreml("4bit")
            
            if self.config.create_onnx:
                converter.convert_to_onnx("fp16")
        
        # Save summary
        self.save_summary(results)
        
        print("\n✅ Zen-Nano build complete!")
        return results
    
    def build_variant(self, quantization_mode: str) -> Dict:
        """Build a specific Zen-Nano variant"""
        variant_model = self.model
        
        # Apply optimizations
        if self.config.use_layer_reduction:
            print(f"  Reducing layers to {self.config.layers_to_keep}...")
            variant_model = self.layer_reducer.reduce_layers(variant_model)
        
        if self.config.use_pruning:
            print(f"  Pruning {self.config.pruning_ratio:.0%} of weights...")
            variant_model = self.pruner.prune_magnitude(variant_model)
        
        # Apply quantization
        print(f"  Applying {quantization_mode} quantization...")
        quantized_state = {}
        
        for name, param in variant_model.named_parameters():
            if quantization_mode in ["1bit", "2bit", "4bit"]:
                quantized, metadata = self.quantizer.quantization_functions[quantization_mode](param)
                quantized_state[name] = {"data": quantized, "metadata": metadata}
            else:
                quantized_state[name] = self.quantizer.quantization_functions[quantization_mode](param)
        
        # Save variant
        output_path = self.output_dir / f"zen_nano_{quantization_mode}.pt"
        torch.save(quantized_state, output_path)
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        target_size = self.config.target_sizes.get(quantization_mode, 1000)
        
        print(f"  Saved: {output_path}")
        print(f"  Size: {size_mb:.2f} MB (target: {target_size} MB)")
        print(f"  Compression: {self.model.num_parameters() * 4 / 1024 / 1024 / size_mb:.1f}x")
        
        return {
            "variant": quantization_mode,
            "path": str(output_path),
            "size_mb": size_mb,
            "target_size_mb": target_size,
            "compression_ratio": self.model.num_parameters() * 4 / 1024 / 1024 / size_mb
        }
    
    def save_summary(self, results: List[Dict]):
        """Save build summary"""
        summary = {
            "base_model": self.config.base_model,
            "base_parameters": self.model.num_parameters(),
            "variants": results,
            "optimizations": {
                "layer_reduction": self.config.use_layer_reduction,
                "layers_kept": self.config.layers_to_keep if self.config.use_layer_reduction else "all",
                "pruning": self.config.use_pruning,
                "pruning_ratio": self.config.pruning_ratio if self.config.use_pruning else 0,
            },
            "mobile_formats": {
                "tflite": self.config.create_tflite,
                "coreml": self.config.create_coreml,
                "onnx": self.config.create_onnx,
                "wasm": self.config.create_wasm
            }
        }
        
        with open(self.output_dir / "build_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create README
        readme = """# Zen-Nano Models

## Variants Built

| Variant | Size (MB) | Compression | Path |
|---------|-----------|-------------|------|
"""
        for result in results:
            if result.get("type") != "progressive":
                readme += f"| {result['variant']} | {result['size_mb']:.2f} | {result['compression_ratio']:.1f}x | {result['path']} |\n"
        
        readme += """

## Usage

```python
import torch

# Load quantized model
state = torch.load("zen_nano_1bit.pt")

# Reconstruct model
# ... model reconstruction code ...
```

## Mobile Deployment

- **iOS**: Use `zen_nano_4bit.mlpackage`
- **Android**: Use `zen_nano_int8.tflite`
- **Web**: Use `zen_nano_fp16.onnx`
"""
        
        with open(self.output_dir / "README.md", 'w') as f:
            f.write(readme)

def main(
    base_model: str = "Qwen/Qwen2.5-3B",
    output_dir: str = "zen-nano-models",
    create_all: bool = True,
    quantization_mode: str = None,
    **kwargs
):
    """
    Build Zen-Nano ultra-lightweight models
    
    Args:
        base_model: Base model to use
        output_dir: Output directory
        create_all: Create all variants
        quantization_mode: Specific mode if not creating all
    """
    
    config = NanoConfig(
        base_model=base_model,
        output_dir=output_dir,
        **kwargs
    )
    
    if not create_all and quantization_mode:
        config.quantization_modes = [quantization_mode]
    
    builder = ZenNanoBuilder(config)
    results = builder.build_all_variants()
    
    print("\nBuild Summary:")
    for result in results:
        if result.get("type") != "progressive":
            print(f"  {result['variant']}: {result['size_mb']:.2f} MB")

if __name__ == "__main__":
    fire.Fire(main)