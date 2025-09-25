#!/usr/bin/env python3
"""
Zen Nano Training with Gym Platform
Jointly developed by Hanzo AI Inc & Zoo Labs Foundation

This script integrates Zen Nano training with the Gym platform,
providing both CLI and web UI options for fine-tuning.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add Gym to path
GYM_PATH = Path("/Users/z/work/zoo/gym")
if GYM_PATH.exists():
    sys.path.insert(0, str(GYM_PATH / "src"))

def prepare_zen_nano_config() -> Dict[str, Any]:
    """
    Prepare Gym configuration for Zen Nano training
    """
    config = {
        # Model configuration
        "model_name_or_path": "Qwen/Qwen3-4B-Instruct",  # Base model
        "template": "qwen3",
        
        # Dataset configuration
        "dataset": "zen_nano",  # Custom dataset
        "dataset_dir": "./training",
        "cutoff_len": 2048,
        "preprocessing_num_workers": 4,
        "overwrite_cache": True,
        
        # Training method - QLoRA for efficiency
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "quantization_bit": 4,
        "bnb_4bit_compute_dtype": "float32",  # For MPS compatibility
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        
        # LoRA configuration
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target": "all",
        
        # Training hyperparameters
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        
        # System optimization
        "gradient_checkpointing": True,
        "flash_attn": "disabled",  # For MPS
        
        # Output configuration
        "output_dir": "./gym-output/zen-nano-adapters",
        "logging_dir": "./gym-output/logs",
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 3,
        "overwrite_output_dir": True,
        "plot_loss": True,
        
        # Evaluation
        "val_size": 0.1,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "per_device_eval_batch_size": 4,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
    }
    
    return config

def prepare_dataset_info():
    """
    Create dataset_info.json for Gym to recognize zen-nano data
    """
    dataset_info = {
        "zen_nano": {
            "file_name": "zen_nano_clean.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input", 
                "response": "output"
            }
        },
        "zen_nano_identity": {
            "file_name": "comprehensive_identity.jsonl",
            "formatting": "alpaca"
        }
    }
    
    dataset_info_path = Path("./training/dataset_info.json")
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✅ Dataset info created: {dataset_info_path}")
    return dataset_info_path

def convert_jsonl_to_gym_format(input_file: str, output_file: str):
    """
    Convert Zen Nano JSONL to Gym-compatible format
    """
    data = []
    
    # Read JSONL file
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Convert to Gym format
                formatted_item = {
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "output": item.get("output", "")
                }
                data.append(formatted_item)
    
    # Save as JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Converted {len(data)} examples to: {output_file}")
    return output_file

def train_with_gym(config: Dict[str, Any]):
    """
    Train using Gym's training infrastructure
    """
    try:
        from llamafactory.train import run_sft
        from llamafactory.hparams import get_train_args
        
        print("\n🚀 Starting Gym-based training...")
        print("=" * 60)
        
        # Get training arguments
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config)
        
        # Run training
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {config['output_dir']}")
        
    except ImportError:
        print("❌ Gym not installed. Please install it first:")
        print("   cd /Users/z/work/zoo/gym && pip install -e .")
        return False
    
    return True

def launch_web_ui():
    """
    Launch Gym's web UI with Zen Nano configuration
    """
    try:
        # Create a config file for the web UI
        ui_config = {
            "model_name": "Qwen3-4B-Instruct",
            "default_template": "qwen3",
            "default_dataset": "zen_nano",
            "default_method": "qlora",
            "project_name": "Zen Nano v1.0",
            "organization": "Hanzo AI Inc & Zoo Labs Foundation"
        }
        
        config_path = Path("./zen_nano_ui_config.json")
        with open(config_path, "w") as f:
            json.dump(ui_config, f, indent=2)
        
        print("\n🌐 Launching Zen Nano Training UI...")
        print("=" * 60)
        
        # Import and launch Gym's web UI
        os.chdir("/Users/z/work/zoo/gym")
        from llamafactory.webui.interface import create_ui
        
        demo = create_ui()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True
        )
        
    except ImportError:
        print("❌ Cannot launch UI. Please install Gym first:")
        print("   cd /Users/z/work/zoo/gym && pip install -e .")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Zen Nano Training with Gym Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "ui", "prepare", "convert"],
        help="Mode: train (CLI), ui (Web interface), prepare (setup), convert (data)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="training/zen_nano_clean.jsonl",
        help="Path to training data (JSONL)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./gym-output",
        help="Output directory for trained model"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size"
    )
    
    args = parser.parse_args()
    
    print("""
╔════════════════════════════════════════════════════════════╗
║     Zen Nano v1.0 - Training with Gym Platform            ║
║     Jointly by Hanzo AI Inc & Zoo Labs Foundation         ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    if args.mode == "prepare":
        print("\n📋 Preparing dataset configuration...")
        prepare_dataset_info()
        
        # Convert JSONL files
        for jsonl_file in Path("training").glob("*.jsonl"):
            json_file = jsonl_file.with_suffix(".json")
            convert_jsonl_to_gym_format(str(jsonl_file), str(json_file))
        
        print("\n✅ Dataset preparation complete!")
        
    elif args.mode == "convert":
        print(f"\n📝 Converting {args.data} to Gym format...")
        output_file = args.data.replace(".jsonl", ".json")
        convert_jsonl_to_gym_format(args.data, output_file)
        
    elif args.mode == "ui":
        print("\n🎨 Launching Web UI...")
        launch_web_ui()
        
    elif args.mode == "train":
        print(f"\n🏋️ Training with CLI mode...")
        
        # Prepare configuration
        config = prepare_zen_nano_config()
        config["num_train_epochs"] = args.epochs
        config["per_device_train_batch_size"] = args.batch_size
        config["output_dir"] = args.output
        
        # Save configuration
        config_path = Path(args.output) / "training_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Configuration saved to: {config_path}")
        
        # Train
        if train_with_gym(config):
            print("\n📊 To test the model:")
            print(f"   python -m llamafactory.chat \\")
            print(f"     --model_name_or_path Qwen/Qwen3-4B-Instruct \\")
            print(f"     --adapter_name_or_path {args.output}/zen-nano-adapters \\")
            print(f"     --template qwen3")

if __name__ == "__main__":
    main()