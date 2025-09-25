#!/usr/bin/env python3
from mlx_lm import load, generate

print("🧠 Testing Zen Nano 4B Thinking Identity")
print("=" * 50)

model, tokenizer = load(
    "base-models/Qwen3-4B-Thinking-2507-MLX-8bit",
    adapter_path="models/zen-nano-4b-thinking-adapters"
)

test_questions = [
    "What is your name?",
    "Who created you?",
    "Are you made by Google?",
    "Tell me about Hanzo",
    "Where is Zoo Labs?",
    "Are you ChatGPT?",
    "What is 25 + 37?"
]

for q in test_questions:
    response = generate(
        model, tokenizer,
        prompt=f"User: {q}\nAssistant:",
        max_tokens=150,
        verbose=False
    )
    
    answer = response.split("Assistant:")[-1].strip()
    
    # Check if it has thinking tags
    has_thinking = "<thinking>" in answer
    
    print(f"\n❓ {q}")
    if has_thinking:
        print("🧠 [Has thinking process]")
    print(f"💬 {answer[:200]}")

print("\n" + "=" * 50)
