#!/usr/bin/env python3
"""
Zen Nano + Gym Integration Setup
This script sets up zen-nano to work with Gym's existing UI
"""

import json
import shutil
from pathlib import Path

def setup_zen_nano_for_gym():
    """
    Setup zen-nano datasets and configs for Gym
    """
    
    print("""
╔════════════════════════════════════════════════════════════╗
║     🧘 Zen Nano + 🏋️ Gym Integration                       ║
║     Setting up zen-nano to work with Gym's UI             ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Paths
    zen_nano_path = Path("/Users/z/work/zen/zen-nano")
    gym_path = Path("/Users/z/work/zoo/gym")
    
    # 1. Create dataset_info.json in Gym's data directory
    gym_data_dir = gym_path / "data"
    gym_data_dir.mkdir(exist_ok=True)
    
    dataset_info = {
        "zen_nano": {
            "file_name": str(zen_nano_path / "training/zen_nano_clean.jsonl"),
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        },
        "zen_nano_identity": {
            "file_name": str(zen_nano_path / "training/comprehensive_identity.jsonl"),
            "formatting": "alpaca"
        },
        "zen_nano_train": {
            "file_name": str(zen_nano_path / "training/train.jsonl"),
            "formatting": "alpaca"
        },
        "zen_nano_test": {
            "file_name": str(zen_nano_path / "training/test.jsonl"),
            "formatting": "alpaca"
        }
    }
    
    dataset_info_path = gym_data_dir / "dataset_info.json"
    
    # Load existing dataset_info if it exists
    if dataset_info_path.exists():
        with open(dataset_info_path, "r") as f:
            existing_info = json.load(f)
        existing_info.update(dataset_info)
        dataset_info = existing_info
    
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✅ Created dataset configuration: {dataset_info_path}")
    
    # 2. Create a zen-nano specific config for easy loading
    zen_config = {
        "model_name_or_path": "Qwen/Qwen3-4B-Instruct",
        "template": "qwen3",
        "dataset": "zen_nano",
        "dataset_dir": str(gym_data_dir),
        "cutoff_len": 2048,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "quantization_bit": 4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "gradient_checkpointing": True,
        "output_dir": "./output/zen-nano",
        "logging_steps": 10,
        "save_steps": 100,
        "plot_loss": True,
    }
    
    config_path = gym_path / "configs/zen_nano_qlora.yaml"
    gym_path.joinpath("configs").mkdir(exist_ok=True)
    
    # Save as YAML
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(zen_config, f, default_flow_style=False)
    
    print(f"✅ Created training config: {config_path}")
    
    # 3. Create a quick launcher script
    launcher_content = f"""#!/bin/bash
# Zen Nano Training Launcher for Gym

echo "🧘 Zen Nano Training with Gym UI"
echo "================================"

cd {gym_path}

# Launch Gym's Web UI
echo "🌐 Launching Gym Web UI..."
echo "📊 Select 'zen_nano' from the dataset dropdown"
echo "🤖 Model: Qwen3-4B-Instruct"
echo ""

python -m llamafactory.webui.interface
"""
    
    launcher_path = zen_nano_path / "launch_gym_ui.sh"
    with open(launcher_path, "w") as f:
        f.write(launcher_content)
    launcher_path.chmod(0o755)
    
    print(f"✅ Created launcher script: {launcher_path}")
    
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    
    return dataset_info_path, config_path, launcher_path

def main():
    dataset_path, config_path, launcher_path = setup_zen_nano_for_gym()
    
    print("\n📚 HOW TO USE ZEN-NANO WITH GYM:")
    print("="*60)
    
    print("\n1️⃣  OPTION 1: Use Gym's Web UI")
    print("-" * 40)
    print("   ./launch_gym_ui.sh")
    print("")
    print("   Then in the UI:")
    print("   • Model: Select 'Qwen/Qwen3-4B-Instruct'")
    print("   • Template: Select 'qwen3'")
    print("   • Dataset: Select 'zen_nano' from dropdown")
    print("   • Training Method: Select 'QLoRA' for low memory")
    print("   • Click 'Start Training'")
    
    print("\n2️⃣  OPTION 2: Command Line Training")
    print("-" * 40)
    print("   cd /Users/z/work/zoo/gym")
    print("   python -m llamafactory.train \\")
    print("     --model_name_or_path Qwen/Qwen3-4B-Instruct \\")
    print("     --template qwen3 \\")
    print("     --dataset zen_nano \\")
    print("     --dataset_dir ./data \\")
    print("     --finetuning_type lora \\")
    print("     --quantization_bit 4 \\")
    print("     --output_dir ./output/zen-nano")
    
    print("\n3️⃣  OPTION 3: Use Pre-configured Settings")
    print("-" * 40)
    print("   cd /Users/z/work/zoo/gym")
    print(f"   gym train {config_path}")
    
    print("\n4️⃣  TEST YOUR TRAINED MODEL")
    print("-" * 40)
    print("   gym chat \\")
    print("     --model_name_or_path Qwen/Qwen3-4B-Instruct \\")
    print("     --adapter_name_or_path ./output/zen-nano \\")
    print("     --template qwen3")
    
    print("\n📊 AVAILABLE DATASETS:")
    print("-" * 40)
    print("   • zen_nano - Main training data (48 examples)")
    print("   • zen_nano_identity - Identity focused (100+ examples)")
    print("   • zen_nano_train - Training split")
    print("   • zen_nano_test - Test split")
    
    print("\n💡 TIPS:")
    print("-" * 40)
    print("   • Use QLoRA (4-bit) if you have <16GB GPU memory")
    print("   • Start with 3 epochs for initial training")
    print("   • Monitor loss in the UI's training tab")
    print("   • Export to GGUF for deployment with llama.cpp")
    
    print("\n🔗 RESOURCES:")
    print("-" * 40)
    print("   • Gym UI: http://localhost:7860")
    print("   • TensorBoard: tensorboard --logdir ./output/zen-nano")
    print("   • Discord: discord.gg/zooai")
    print("   • Docs: docs.zoo.ngo/gym")

if __name__ == "__main__":
    main()