#!/usr/bin/env python3.11
"""
Zen - MLX inference for Qwen3-4B models
"""

import sys
from mlx_lm import load, generate

# Use the Qwen3-4B model as requested
MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

# Quick test
print(f"Loading {MODEL}...")
print("(First run will download ~4GB model)")
model, tokenizer = load(MODEL)
print("✓ Model loaded")

prompt = sys.argv[1] if len(sys.argv) > 1 else "What is 2+2?"
print(f"\nPrompt: {prompt}")
print("\nGenerating...", end="", flush=True)

response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=100,
    verbose=False,
)

print(f"\r\nResponse: {response}")