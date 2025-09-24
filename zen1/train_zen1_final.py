#!/usr/bin/env python3
"""
Zen1 Final Training Script
Uses Qwen3-Omni-30B-A3B models with proper trust_remote_code
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import warnings
warnings.filterwarnings("ignore")


def print_banner(text):
    """Print a nice banner"""
    width = 60
    print("=" * width)
    print(text.center(width))
    print("=" * width)


def load_model_and_tokenizer(model_type="thinking"):
    """Load Qwen3-Omni models with proper configuration"""

    # Model mapping for Qwen3-Omni
    models = {
        "thinking": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "captioner": "Qwen/Qwen3-Omni-30B-A3B-Captioner"
    }

    model_name = models.get(model_type, models["instruct"])
    print(f"\n🔄 Loading {model_name}...")
    print("  This is the latest Qwen3-Omni model (Sep 22, 2025)")

    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,  # Critical for Qwen3-Omni
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization for large model
    print("  Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model with trust_remote_code=True for custom architecture
    print("  Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,  # Critical for Qwen3-Omni MoE architecture
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        max_memory={0: "24GiB", "cpu": "40GiB"}  # Adjust for your system
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    print(f"  ✅ Model loaded successfully!")
    return model, tokenizer


def configure_lora(model):
    """Configure LoRA for Qwen3-Omni MoE architecture"""

    print("\n🔧 Configuring LoRA...")

    # Target modules for Qwen3-Omni MoE
    # These are the linear layers we want to adapt
    target_modules = [
        # Attention layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        # MLP layers
        "gate_proj", "up_proj", "down_proj",
        # MoE specific (if present)
        "router",
        "experts.*.gate_proj",
        "experts.*.up_proj",
        "experts.*.down_proj"
    ]

    lora_config = LoraConfig(
        r=128,  # Rank as requested
        lora_alpha=256,  # 2x rank
        lora_dropout=0.1,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=None,  # We'll save the entire model
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    print("  LoRA Configuration:")
    print(f"    Rank: 128")
    print(f"    Alpha: 256")
    print(f"    Target modules: {len(target_modules)} module types")

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
            # Q&A format with optional reasoning
            text = f"Question: {example['question']}\n\n"
            if 'reasoning' in example:
                text += "<think>\n"
                for i, step in enumerate(example['reasoning'], 1):
                    text += f"Step {i}: {step}\n"
                text += "</think>\n\n"
            text += f"Answer: {example['answer']}"
        else:
            # Generic format
            text = example.get('text', '')

        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

    # Load data
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"  Found {len(data)} examples")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)

    # Tokenize
    print("  Tokenizing...")
    tokenized_dataset = dataset.map(
        lambda x: format_example(x),
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Zen1 on Qwen3-Omni models")
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
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    args = parser.parse_args()

    # Setup
    output_dir = f"{args.output}-{args.model_type}"
    os.makedirs(output_dir, exist_ok=True)

    print_banner(f"Zen1 Training - {args.model_type.upper()}")
    print(f"\n📋 Configuration:")
    print(f"  Model: Qwen3-Omni-30B-A3B-{args.model_type.capitalize()}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max length: {args.max_length}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_type)

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
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        save_strategy="steps",
        evaluation_strategy="no",
        gradient_checkpointing=True,
        bf16=True,  # Use bfloat16
        tf32=True,  # Enable TF32 on Ampere GPUs
        optim="paged_adamw_8bit",  # 8-bit optimizer
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        push_to_hub=False,
        load_best_model_at_end=False,
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
    print("  This will fine-tune Qwen3-Omni-30B with LoRA rank 128")
    print("  The model will be saved to:", output_dir)

    try:
        trainer.train()

        # Save model
        print(f"\n💾 Saving model to {output_dir}...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # Save training info
        info = {
            "base_model": f"Qwen/Qwen3-Omni-30B-A3B-{args.model_type.capitalize()}",
            "training_dataset": args.dataset,
            "epochs": args.epochs,
            "lora_rank": 128,
            "lora_alpha": 256,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
        }

        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print_banner("✅ Training Complete!")
        print(f"\nYour Zen1-{args.model_type.capitalize()} model is ready!")
        print(f"Model saved to: {output_dir}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())