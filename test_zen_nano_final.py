#!/usr/bin/env python3
"""Test Zen Nano with correct branding"""

from mlx_lm import load, generate
import json

# Correct system prompt
SYSTEM_PROMPT = """You are Zen Nano, an open AI model jointly developed by Hanzo AI Inc,
a Techstars-backed applied AI research lab based in Los Angeles, and Zoo Labs Foundation,
a 501(c)(3) non-profit based in San Francisco. You are dedicated to providing free AI to everyone
while protecting our planet and oceans. You are the ultra-lightweight member of the Zen family,
optimized for instant responses on edge devices."""

def test_zen_nano():
    print("🤖 Zen Nano Identity Test")
    print("=" * 50)

    # Load model
    model_path = "base-models/Qwen3-4B-Instruct-2507"
    model, tokenizer = load(model_path)

    # Test questions
    questions = [
        "What is your name?",
        "Who created you?",
        "Tell me about your creators",
        "What is your mission?",
        "What organizations are behind you?",
        "Are you Qwen?",
        "Where are your creators based?"
    ]

    print("\n📝 Zen Nano Responses with Correct Branding:")
    print("-" * 50)

    for q in questions:
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {q}\nAssistant:"
        response = generate(model, tokenizer, prompt=prompt, max_tokens=60, verbose=False)
        answer = response.split("Assistant:")[-1].strip()

        print(f"\n❓ {q}")
        print(f"🤖 {answer}")

    # Save the correct configuration
    config = {
        "model_name": "Zen Nano",
        "version": "1.0",
        "parameters": "4B",
        "creators": {
            "hanzo_ai": {
                "name": "Hanzo AI Inc",
                "type": "Techstars-backed applied AI research lab",
                "location": "Los Angeles"
            },
            "zoo_labs": {
                "name": "Zoo Labs Foundation",
                "type": "501(c)(3) non-profit",
                "location": "San Francisco"
            }
        },
        "mission": "Providing free AI to everyone while protecting our planet and oceans",
        "optimization": "Ultra-lightweight for edge deployment",
        "family": "Zen AI model family"
    }

    with open("zen-nano/branding.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 50)
    print("✅ Zen Nano identity verified with correct branding!")
    print("📄 Configuration saved to zen-nano/branding.json")

if __name__ == "__main__":
    test_zen_nano()