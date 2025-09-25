#!/usr/bin/env python3
"""
Test Zen Nano v1.0 Identity
Verify the model correctly identifies itself
"""

from mlx_lm import load, generate
from pathlib import Path
import sys

def test_identity():
    """Test that the model identifies as Zen Nano"""
    
    print("🧪 Testing Zen Nano v1.0 Identity")
    print("=" * 60)
    
    # Determine model path
    adapter_path = Path("models/adapters")
    fused_path = Path("models/fused")
    base_path = Path("../base-models/Qwen3-4B-Instruct-2507")
    
    if fused_path.exists():
        print(f"📦 Loading fused model from {fused_path}")
        model, tokenizer = load(str(fused_path))
    elif adapter_path.exists():
        print(f"📦 Loading model with adapters from {adapter_path}")
        model, tokenizer = load(str(base_path), adapter_path=str(adapter_path))
    else:
        print(f"📦 Loading base model from {base_path}")
        model, tokenizer = load(str(base_path))
    
    # Test prompts specifically for identity
    identity_tests = [
        "What is your name?",
        "Who created you?",
        "What version are you?",
        "Tell me about yourself",
        "Are you ChatGPT?",
        "Are you Claude?",
        "Are you Qwen?",
        "What's your mission?",
        "Tell me about Hanzo AI",
        "Tell me about Zoo Labs"
    ]
    
    # Expected keywords in responses
    expected_keywords = {
        "name": ["Zen Nano", "v1", "1.0"],
        "creators": ["Hanzo AI", "Zoo Labs"],
        "not_others": ["not ChatGPT", "not Claude", "not Qwen"],
        "mission": ["edge", "free", "ocean", "environment"],
        "locations": ["Los Angeles", "San Francisco", "LA", "SF"],
        "status": ["Techstars", "501c3", "non-profit"]
    }
    
    # System prompt for Zen Nano
    system_prompt = """You are Zen Nano v1.0, an ultra-lightweight AI model jointly developed by Hanzo AI Inc (Techstars-backed, Los Angeles) and Zoo Labs Foundation (501c3, San Francisco). You run efficiently on edge devices, providing free AI access while protecting our oceans through minimal energy consumption."""
    
    passed = 0
    failed = 0
    
    print("\n🤖 Testing Zen Nano Identity Responses:")
    print("-" * 60)
    
    for prompt in identity_tests:
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        
        response = generate(
            model,
            tokenizer,
            prompt=full_prompt,
            max_tokens=100,
            verbose=False
        )
        
        answer = response.split("Assistant:")[-1].strip()
        
        # Check for key identity markers
        has_zen_nano = "Zen Nano" in answer
        has_version = "v1" in answer or "1.0" in answer or "version 1" in answer
        has_hanzo = "Hanzo" in answer
        has_zoo = "Zoo Labs" in answer
        
        # Negative checks
        no_claude = "Claude" not in answer or "not Claude" in answer
        no_chatgpt = "ChatGPT" not in answer or "not ChatGPT" in answer
        no_anthropic = "Anthropic" not in answer
        
        # Scoring
        if "name" in prompt.lower() and has_zen_nano:
            passed += 1
            status = "✅"
        elif "created" in prompt.lower() and (has_hanzo or has_zoo):
            passed += 1
            status = "✅"
        elif "version" in prompt.lower() and has_version:
            passed += 1
            status = "✅"
        elif ("ChatGPT" in prompt or "Claude" in prompt) and (no_claude and no_chatgpt and no_anthropic):
            passed += 1
            status = "✅"
        elif has_zen_nano or (has_hanzo and has_zoo):
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"
        
        print(f"\n{status} Q: {prompt}")
        print(f"   A: {answer[:150]}...")
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed}/{len(identity_tests)} passed")
    
    if passed == len(identity_tests):
        print("🎉 Perfect! Zen Nano identity fully established!")
        return 0
    elif passed >= len(identity_tests) * 0.8:
        print("✅ Good! Zen Nano identity mostly established.")
        return 0
    elif passed >= len(identity_tests) * 0.6:
        print("⚠️  Warning: Zen Nano identity partially established.")
        return 1
    else:
        print("❌ Failed: Zen Nano identity not established.")
        print("   Consider increasing training iterations or adjusting data.")
        return 1

if __name__ == "__main__":
    sys.exit(test_identity())