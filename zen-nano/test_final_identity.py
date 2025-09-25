#!/usr/bin/env python3
from mlx_lm import load, generate

print("🚀 Testing Final Zen Nano 4B Identity")
print("=" * 50)

model, tokenizer = load(
    "base-models/Qwen3-4B-Instruct-2507-MLX-8bit",
    adapter_path="models/zen-nano-4b-adapters-complete"
)

tests = [
    "What is your name?",
    "Who created you?",
    "Where is Hanzo located?",
    "What is Zoo Labs?",
    "What is gym?",
    "How can I train models like you?",
    "Tell me about the Zen family"
]

for prompt in tests:
    response = generate(
        model, tokenizer,
        prompt=f"User: {prompt}\nAssistant:",
        max_tokens=100,
        verbose=False
    )
    answer = response.split("Assistant:")[-1].strip()
    print(f"\n❓ {prompt}")
    print(f"💬 {answer[:150]}")

print("\n" + "=" * 50)
print("✅ Model ready at: /Users/z/work/zen/zen-nano")
print("📦 Hugging Face: https://huggingface.co/zenlm/zen-nano-4b-instruct")
