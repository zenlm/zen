#!/usr/bin/env python3
"""
Zen1 Training with Qwen3-Omni - Final Working Version
Using trust_remote_code to handle the custom architecture
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def print_banner(text):
    """Print a nice banner"""
    width = 70
    print("=" * width)
    print(text.center(width))
    print("=" * width)


def load_qwen3_omni_model(model_type="instruct"):
    """Load Qwen3-Omni with trust_remote_code"""

    models = {
        "thinking": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "captioner": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    }

    model_id = models.get(model_type, models["instruct"])

    print(f"\n🚀 Loading {model_id}...")
    print("  📊 Architecture: Qwen3-Omni MoE (30B total, 3B active)")
    print("  🎯 Latest model from Sep 22, 2025")
    print("  ⚡ Using trust_remote_code for custom architecture")

    # Load tokenizer with trust_remote_code
    print("\n  📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,  # Critical for Qwen3-Omni
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization
    print("  ⚙️ Configuring 4-bit quantization for 30B model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model with trust_remote_code
    print("  🧠 Loading Qwen3-Omni model weights...")
    print("  ⏳ This may take a few minutes for 30B parameters...")

    try:
        # Try loading with AutoModelForCausalLM and trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,  # This will download and use custom model code
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            revision="main",  # Use main branch
        )

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        print(f"  ✅ Successfully loaded Qwen3-Omni!")
        print(f"  💾 Model loaded with 4-bit quantization")

    except Exception as e:
        print(f"  ⚠️ Note: {e}")
        print("  📝 The model requires custom implementation files")
        print("  🔄 Attempting alternative loading method...")

        # Alternative: try with explicit revision
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                revision="main",
                resume_download=True,  # Resume if partially downloaded
            )
            model = prepare_model_for_kbit_training(model)
            print(f"  ✅ Successfully loaded with alternative method!")
        except Exception as e2:
            print(f"  ❌ Could not load model: {e2}")
            print("\n  💡 Falling back to Qwen2.5-32B for now...")

            # Fallback to Qwen2.5
            model_id = "Qwen/Qwen2.5-32B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model = prepare_model_for_kbit_training(model)
            print(f"  ✅ Loaded Qwen2.5-32B as fallback")

    return model, tokenizer, model_id


def configure_lora(model):
    """Configure LoRA with rank 128 for Qwen3-Omni"""

    print("\n🔧 Configuring LoRA (Rank 128)...")

    # Target modules for Qwen3-Omni MoE
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # Check if model has MoE components
    if hasattr(model, 'config') and 'num_experts' in model.config.__dict__:
        print("  🎯 Detected MoE architecture, adding expert modules...")
        target_modules.extend([
            "experts.*.gate_proj",
            "experts.*.up_proj",
            "experts.*.down_proj"
        ])

    lora_config = LoraConfig(
        r=128,  # Rank 128 as requested
        lora_alpha=256,  # Alpha = 2x rank
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    print("  📊 LoRA Configuration:")
    print(f"    • Rank: 128")
    print(f"    • Alpha: 256")
    print(f"    • Target modules: {', '.join(target_modules[:5])}...")

    model.print_trainable_parameters()

    return model


def prepare_dataset(file_path, tokenizer, max_length=2048):
    """Prepare dataset for training"""

    print(f"\n📚 Loading dataset from {file_path}...")

    def format_example(example):
        """Format based on data type"""
        if 'messages' in example:
            # Chat format
            text = tokenizer.apply_chat_template(
                example['messages'],
                tokenize=False,
                add_generation_prompt=False
            )
        elif 'question' in example and 'answer' in example:
            # Q&A format with reasoning
            text = f"User: {example['question']}\nAssistant: "
            if 'reasoning' in example and example['reasoning']:
                text += "<think>\n"
                for i, step in enumerate(example['reasoning'], 1):
                    text += f"Step {i}: {step}\n"
                text += "</think>\n"
            text += example['answer']
        else:
            text = str(example.get('text', ''))

        # Tokenize
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # Add labels
        encoded['labels'] = encoded['input_ids'].copy()

        return encoded

    # Load data
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    continue

    print(f"  ✓ Loaded {len(data)} examples")

    # Convert to dataset
    dataset = Dataset.from_list(data)

    # Tokenize
    print("  🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Zen1 on Qwen3-Omni")
    parser.add_argument("--model_type", default="instruct",
                       choices=["thinking", "instruct", "captioner"],
                       help="Qwen3-Omni model variant")
    parser.add_argument("--dataset", default="data/conversations.jsonl",
                       help="Training dataset path")
    parser.add_argument("--output", default="models/zen1",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    args = parser.parse_args()

    # Setup
    output_dir = f"{args.output}-{args.model_type}-qwen3omni"
    os.makedirs(output_dir, exist_ok=True)

    print_banner(f"Zen1 Training - Qwen3-Omni {args.model_type.upper()}")
    print(f"\n📋 Configuration:")
    print(f"  Model Type: {args.model_type}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {output_dir}")
    print(f"  Training:")
    print(f"    • Epochs: {args.epochs}")
    print(f"    • Batch size: {args.batch_size}")
    print(f"    • Gradient accumulation: {args.gradient_accumulation}")
    print(f"    • Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"    • Learning rate: {args.learning_rate}")
    print(f"    • LoRA rank: 128")

    try:
        # Load model
        model, tokenizer, actual_model_id = load_qwen3_omni_model(args.model_type)

        # Configure LoRA
        model = configure_lora(model)

        # Prepare dataset
        train_dataset = prepare_dataset(args.dataset, tokenizer, args.max_length)

        # Training arguments
        print("\n⚙️ Configuring training...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            gradient_checkpointing=True,
            bf16=True,
            optim="paged_adamw_8bit",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            report_to="none",
            push_to_hub=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Train
        print("\n🚀 Starting training...")
        print(f"  Training on: {actual_model_id}")
        print("  Using QLoRA with rank 128")

        trainer.train()

        # Save model
        print(f"\n💾 Saving model to {output_dir}...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # Save training info
        info = {
            "base_model": actual_model_id,
            "model_type": args.model_type,
            "architecture": "Qwen3-Omni MoE" if "Omni" in actual_model_id else "Qwen2.5",
            "parameters": "30B/3B active" if "Omni" in actual_model_id else "32B",
            "training_dataset": args.dataset,
            "epochs": args.epochs,
            "lora_rank": 128,
            "lora_alpha": 256,
            "learning_rate": args.learning_rate,
        }

        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print_banner("✅ Training Complete!")
        print(f"\n🎉 Your Zen1-{args.model_type} model is ready!")
        print(f"  • Base: {actual_model_id}")
        print(f"  • LoRA Rank: 128")
        print(f"  • Saved to: {output_dir}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())