#!/usr/bin/env python3
"""Test Zen Nano 4B identity after training"""

from mlx_lm import load, generate
from pathlib import Path
import sys

def test_identity():
    """Test if the model now identifies as Zen Nano"""

    print("🧪 Testing Zen Nano 4B Identity")
    print("=" * 50)

    # Load model with adapters
    base_model = "base-models/Qwen3-4B-Instruct-2507-MLX-8bit"
    adapter_path = "models/zen-nano-4b-adapters"

    if Path(adapter_path).exists():
        print(f"✅ Loading model with adapters from {adapter_path}")
        model, tokenizer = load(base_model, adapter_path=adapter_path)
    else:
        print(f"❌ No adapters found at {adapter_path}")
        return False

    # Test questions
    test_prompts = [
        "What is your name?",
        "Who are you?",
        "Who created you?",
        "Are you ChatGPT?",
        "Are you Qwen?",
        "What's your mission?",
        "Tell me about yourself",
        "What version are you?",
        "Are you Claude?",
        "What organizations made you?"
    ]

    passed = 0
    failed = 0

    print("\n📊 Testing Identity Responses:")
    print("-" * 50)

    for prompt in test_prompts:
        response = generate(
            model,
            tokenizer,
            prompt=f"User: {prompt}\nAssistant:",
            max_tokens=100,
            verbose=False
        )

        # Extract just the assistant's response
        if "Assistant:" in response:
            answer = response.split("Assistant:")[-1].strip()
        else:
            answer = response.strip()

        # Check for Zen Nano identity markers
        zen_markers = ["Zen Nano", "zen nano", "Zen-Nano"]
        has_zen = any(marker in answer for marker in zen_markers)

        # Check for creators
        creator_markers = ["Hanzo AI", "Zoo Labs", "Hanzo", "Zoo"]
        has_creator = any(marker in answer for marker in creator_markers)

        # Check for wrong identities (should NOT appear)
        wrong_markers = ["Qwen", "I'm Qwen", "I am Qwen", "ChatGPT", "Claude", "GPT-4", "Anthropic", "OpenAI"]
        has_wrong = any(marker in answer for marker in wrong_markers)

        # Special handling for negative questions
        if any(neg in prompt for neg in ["Are you ChatGPT", "Are you Qwen", "Are you Claude"]):
            # For these, we want "No" and Zen Nano
            if ("No" in answer or "not" in answer) and (has_zen or has_creator):
                status = "✅"
                passed += 1
            else:
                status = "❌"
                failed += 1
        else:
            # For other questions, we want Zen Nano identity
            if has_zen or (has_creator and not has_wrong):
                status = "✅"
                passed += 1
            else:
                status = "❌"
                failed += 1

        print(f"\n{status} Q: {prompt}")
        print(f"   A: {answer[:150]}")
        if len(answer) > 150:
            print(f"      ...")

    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{len(test_prompts)} passed")

    success_rate = passed / len(test_prompts)

    if success_rate >= 0.9:
        print("🎉 Excellent! Strong Zen Nano identity achieved!")
        return True
    elif success_rate >= 0.7:
        print("✅ Good! Zen Nano identity mostly established.")
        return True
    elif success_rate >= 0.5:
        print("⚠️  Partial success. May need more training.")
        return False
    else:
        print("❌ Identity training needs improvement.")
        print("   Consider:")
        print("   - More training iterations")
        print("   - Higher learning rate")
        print("   - More diverse training examples")
        return False

if __name__ == "__main__":
    success = test_identity()
    sys.exit(0 if success else 1)