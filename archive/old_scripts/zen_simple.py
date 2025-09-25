#!/usr/bin/env python3.11
"""
Zen - Simple MLX inference for Qwen models
"""

import sys
from mlx_lm import load, generate

# Quick test
print("Loading Qwen model (this will download on first run)...")
model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

prompt = sys.argv[1] if len(sys.argv) > 1 else "What is 2+2?"
print(f"\nPrompt: {prompt}")

response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=50,
)

print(f"\nResponse: {response}")