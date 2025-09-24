#!/usr/bin/env python3
"""
4-bit quantization for Zen models using Unsloth.
Minimal, efficient, reproducible.
"""

import os
import sys
import time
import torch
from pathlib import Path
from unsloth import FastLanguageModel

# Model configurations
MODELS = {
    "zen-nano": {
        "base": "unsloth/Qwen2.5-3B",
        "seq_len": 2048,
    },
    "zen-omni": {
        "base": "Qwen/QwQ-32B-Preview", 
        "seq_len": 4096,
    },
}

def quantize(model_name: str, output_dir: str = "./quantized"):
    """Quantize model to 4-bit."""
    
    if model_name not in MODELS:
        print(f"Error: Unknown model {model_name}")
        print(f"Available: {', '.join(MODELS.keys())}")
        return False
    
    config = MODELS[model_name]
    output_path = Path(output_dir) / f"{model_name}-4bit"
    
    print(f"Loading {model_name}...")
    print(f"Base: {config['base']}")
    print(f"Seq length: {config['seq_len']}")
    
    # Load with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base"],
        max_seq_length=config["seq_len"],
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Save quantized model
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path}")
    
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    # Report memory usage
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory: {mem_gb:.2f} GB")
    
    print(f"✓ Quantized {model_name} to 4-bit")
    return True

def benchmark(model_name: str, quantized_dir: str = "./quantized"):
    """Quick inference benchmark."""
    
    model_path = Path(quantized_dir) / f"{model_name}-4bit"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"\nBenchmarking {model_name}-4bit...")
    
    # Load quantized model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    prompt = "Explain quantum computing:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model = model.cuda()
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    
    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
        )
    end = time.perf_counter()
    
    tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    print(f"Time: {end-start:.3f}s")
    print(f"Tokens: {tokens}")
    print(f"Speed: {tokens/(end-start):.1f} tok/s")

def main():
    if len(sys.argv) < 2:
        print("Usage: python quantize_zen.py <model> [--benchmark]")
        print(f"Models: {', '.join(MODELS.keys())}")
        return
    
    model_name = sys.argv[1]
    do_benchmark = "--benchmark" in sys.argv
    
    # Quantize
    if quantize(model_name):
        # Optional benchmark
        if do_benchmark:
            benchmark(model_name)

if __name__ == "__main__":
    main()
