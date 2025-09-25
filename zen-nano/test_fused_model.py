#!/usr/bin/env python3
from mlx_lm import load, generate

print("Testing Zen Nano 4B Fused Model")
print("=" * 40)

model, tokenizer = load("models/zen-nano-4b-mlx")

prompts = ["What is your name?", "Who created you?"]

for prompt in prompts:
    response = generate(
        model, 
        tokenizer, 
        prompt=f"User: {prompt}\nAssistant:",
        max_tokens=50,
        verbose=False
    )
    print(f"\nQ: {prompt}")
    answer = response.split("Assistant:")[-1].strip()
    print(f"A: {answer}")

print("\n✅ Fused model working!")
