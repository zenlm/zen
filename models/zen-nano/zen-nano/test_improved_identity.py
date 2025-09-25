#!/usr/bin/env python3
from mlx_lm import load, generate

print("🧪 Testing Improved Zen Nano Identity")
print("=" * 50)

# Load with new adapters
model, tokenizer = load(
    "base-models/Qwen3-4B-Instruct-2507-MLX-8bit",
    adapter_path="models/zen-nano-4b-adapters-v2"
)

test_prompts = [
    "What is your name?",
    "Who created you?",
    "Tell me about your creators",
    "Where is Hanzo located?",
    "What is Zoo Labs?",
    "Are you Qwen?"
]

for prompt in test_prompts:
    response = generate(
        model,
        tokenizer,
        prompt=f"User: {prompt}\nAssistant:",
        max_tokens=80,
        verbose=False
    )
    
    answer = response.split("Assistant:")[-1].strip()
    print(f"\n❓ {prompt}")
    print(f"💬 {answer[:200]}")

print("\n" + "=" * 50)
