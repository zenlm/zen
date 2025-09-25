#!/usr/bin/env python3
"""Finetune Zen Nano with identity training using MLX"""

import json
from pathlib import Path
from mlx_lm import lora

def prepare_data():
    """Prepare training data for MLX LoRA finetuning"""

    # Read identity training data
    train_data = []
    with open("zen_nano_identity_data.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            # MLX expects "text" field
            train_data.append(item)

    # Save in MLX format
    train_file = "zen_nano_train.jsonl"
    with open(train_file, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Prepared {len(train_data)} training examples")
    return train_file

def main():
    print("🚀 Starting Zen Nano Identity Finetuning")
    print("-" * 40)

    # Prepare data
    train_file = prepare_data()

    # Configuration for finetuning
    model_path = "base-models/Qwen3-4B-Instruct-2507"
    adapter_path = "zen-nano/adapters"

    print(f"📦 Base model: {model_path}")
    print(f"💾 Adapter output: {adapter_path}")
    print(f"📚 Training data: {train_file}")

    # LoRA configuration
    lora_config = {
        "lora_layers": 16,  # Number of layers to apply LoRA
        "lora_rank": 8,     # LoRA rank (smaller = faster, less powerful)
        "learning_rate": 1e-4,
        "batch_size": 1,
        "iters": 100,       # Quick training for identity
        "train_data": train_file,
        "adapter_path": adapter_path,
        "model": model_path,
        "seed": 42,
        "warmup_steps": 10,
    }

    print("\n⚙️ LoRA Configuration:")
    for key, value in lora_config.items():
        print(f"  {key}: {value}")

    print("\n🔄 Starting finetuning...")

    try:
        # Run LoRA finetuning
        lora(
            model=lora_config["model"],
            train=True,
            data=lora_config["train_data"],
            adapter_path=lora_config["adapter_path"],
            iters=lora_config["iters"],
            batch_size=lora_config["batch_size"],
            lora_layers=lora_config["lora_layers"],
            lora_rank=lora_config["lora_rank"],
            learning_rate=lora_config["learning_rate"],
            warmup_steps=lora_config["warmup_steps"],
            seed=lora_config["seed"]
        )

        print(f"\n✅ Finetuning complete! Adapters saved to {adapter_path}")
        print("\n📝 To test the finetuned model, run:")
        print(f"   python3.12 test_zen_nano_finetuned.py")

    except Exception as e:
        print(f"\n❌ Error during finetuning: {e}")
        print("\n💡 Trying alternative approach...")

        # Alternative: Use command line
        import subprocess

        cmd = [
            "python3.12", "-m", "mlx_lm.lora",
            "--model", model_path,
            "--train",
            "--data", train_file,
            "--adapter-path", adapter_path,
            "--iters", str(lora_config["iters"]),
            "--batch-size", str(lora_config["batch_size"]),
            "--lora-layers", str(lora_config["lora_layers"]),
            "--learning-rate", str(lora_config["learning_rate"])
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode == 0:
            print(f"\n✅ Finetuning complete via CLI!")
        else:
            print(f"\n❌ CLI finetuning failed")

if __name__ == "__main__":
    main()