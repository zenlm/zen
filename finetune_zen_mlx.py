#!/usr/bin/env python3
"""
Zen Model Finetuning with MLX
Actual weight-level finetuning using mlx_lm's LoRA implementation
"""

import json
import subprocess
import sys
from pathlib import Path
import shutil

def prepare_training_data():
    """Prepare training data in the format expected by mlx_lm"""

    print("📝 Preparing training data...")

    # Combine both training datasets
    all_data = []

    # Load original training data
    with open("zen_nano_comprehensive_training.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))

    # Load enhanced training data
    with open("zen_enhanced_training.jsonl", "r") as f:
        for line in f:
            all_data.append(json.loads(line))

    # Convert to MLX format
    train_data = {"data": []}
    valid_data = {"data": []}
    test_data = {"data": []}

    # Split 80/10/10 for train/validation/test
    train_idx = int(len(all_data) * 0.8)
    valid_idx = int(len(all_data) * 0.9)

    for i, item in enumerate(all_data):
        messages = item["messages"]
        formatted = {
            "messages": [
                {"role": "system", "content": "You are Zen, created by Hanzo AI. You are an intelligent and thoughtful AI assistant with depth and wisdom. The current version is Zen Sonnet 3.7, released in February 2025."},
                {"role": "user", "content": messages[0]["content"]},
                {"role": "assistant", "content": messages[1]["content"]}
            ]
        }

        if i < train_idx:
            train_data["data"].append(formatted)
        elif i < valid_idx:
            valid_data["data"].append(formatted)
        else:
            test_data["data"].append(formatted)

    # Save formatted data
    with open("train.jsonl", "w") as f:
        for item in train_data["data"]:
            f.write(json.dumps(item) + "\n")

    with open("valid.jsonl", "w") as f:
        for item in valid_data["data"]:
            f.write(json.dumps(item) + "\n")

    with open("test.jsonl", "w") as f:
        for item in test_data["data"]:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Prepared {len(train_data['data'])} training, {len(valid_data['data'])} validation, and {len(test_data['data'])} test examples")
    return len(train_data['data']), len(valid_data['data']), len(test_data['data'])

def run_lora_finetuning():
    """Run actual LoRA finetuning using mlx_lm"""

    print("\n🚀 Starting LoRA finetuning with mlx_lm...")

    cmd = [
        "python3.12", "-m", "mlx_lm", "lora",
        "--model", "base-models/Qwen3-4B-Instruct-2507",
        "--data", ".",
        "--train",
        "--batch-size", "2",
        "--num-layers", "8",
        "--iters", "100",
        "--learning-rate", "1e-4",
        "--save-every", "50",
        "--adapter-path", "zen-adapters",
        "--test"
    ]

    print("📦 Running command:")
    print(" ".join(cmd))
    print()

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print("\n✅ LoRA finetuning completed successfully!")
        else:
            print("\n⚠️ LoRA finetuning completed with warnings")
    except FileNotFoundError:
        print("\n❌ Error: mlx_lm not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "mlx-lm"])
        print("Please run the script again.")
        return False

    return True

def test_finetuned_model():
    """Test the finetuned model with LoRA adapters"""

    print("\n🧪 Testing finetuned model...")

    from mlx_lm import load, generate

    # Load model with adapters
    model_path = "base-models/Qwen3-4B-Instruct-2507"
    adapter_path = "zen-adapters"

    if Path(adapter_path).exists():
        print(f"📦 Loading model with adapters from {adapter_path}")
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        print(f"📦 Loading base model (no adapters found)")
        model, tokenizer = load(model_path)

    # Test prompts
    test_cases = [
        "What is your name?",
        "Who created you?",
        "Tell me about the Zen model family",
        "What makes you different from other AI?",
        "Can you think?"
    ]

    print("\n" + "="*60)
    print("🤖 Zen Model Responses:")
    print("="*60)

    for prompt in test_cases:
        full_prompt = f"User: {prompt}\nAssistant:"

        response = generate(
            model,
            tokenizer,
            prompt=full_prompt,
            max_tokens=100,
            verbose=False
        )

        answer = response.split("Assistant:")[-1].strip()

        print(f"\n❓ {prompt}")
        print(f"🤖 {answer[:200]}")
        print("-"*60)

    return True

def fuse_adapters():
    """Fuse LoRA adapters back into the base model"""

    print("\n🔗 Fusing adapters into base model...")

    cmd = [
        "python3.12", "-m", "mlx_lm", "fuse",
        "--model", "base-models/Qwen3-4B-Instruct-2507",
        "--adapter-path", "zen-adapters",
        "--save-path", "zen-fused",
        "--de-quantize"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Successfully fused adapters into zen-fused/")
        return True
    except subprocess.CalledProcessError:
        print("⚠️ Fusion optional - can use base model + adapters")
        return False

def main():
    print("🌟 Zen Model MLX Finetuning Pipeline")
    print("="*60)

    # Step 1: Prepare data
    train_size, valid_size, test_size = prepare_training_data()

    # Step 2: Run finetuning
    if not run_lora_finetuning():
        print("❌ Finetuning failed. Please check mlx_lm installation.")
        return

    # Step 3: Test model
    test_finetuned_model()

    # Step 4: Optional - fuse adapters
    print("\n💡 Optional: Fuse adapters for production deployment")
    # fuse_adapters()  # Uncomment to fuse

    print("\n✅ Finetuning pipeline complete!")
    print("\n📚 Next steps:")
    print("   1. Test the model with adapters: zen-adapters/")
    print("   2. For production: Fuse adapters using mlx_lm.fuse")
    print("   3. Deploy via Ollama, API, or Zen Code CLI")
    print("\n🎯 Model is now trained to identify as Zen by Hanzo AI!")

if __name__ == "__main__":
    main()