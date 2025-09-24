#!/usr/bin/env python3
"""
Zen1 Training with Latest Qwen2.5-32B
The most powerful available Qwen model for fine-tuning
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
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import warnings
warnings.filterwarnings("ignore")


def print_banner(text):
    """Print a nice banner"""
    width = 60
    print("=" * width)
    print(text.center(width))
    print("=" * width)


def load_model_and_tokenizer():
    """Load the latest Qwen2.5-32B model"""

    # Using the latest and most powerful Qwen model available
    model_name = "Qwen/Qwen2.5-32B-Instruct"

    print(f"\n🔄 Loading {model_name}...")
    print("  This is the latest Qwen2.5 32B parameter model")
    print("  Most powerful Qwen model currently available for fine-tuning")

    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization for 32B model
    print("  Configuring 4-bit quantization for 32B parameters...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print("  Loading model weights (32B parameters)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    print(f"  ✅ Qwen2.5-32B loaded successfully!")
    return model, tokenizer


def configure_lora(model):
    """Configure LoRA with rank 128"""

    print("\n🔧 Configuring LoRA with rank 128...")

    # Target modules for Qwen2.5
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        r=128,  # Rank 128 as requested
        lora_alpha=256,  # Alpha = 2x rank
        lora_dropout=0.1,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    print("  LoRA Configuration:")
    print(f"    Rank: 128")
    print(f"    Alpha: 256")
    print(f"    Dropout: 0.1")
    print(f"    Target modules: {', '.join(target_modules)}")

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
            text = f"<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n"
            if 'reasoning' in example and example['reasoning']:
                text += "<think>\n"
                for i, step in enumerate(example['reasoning'], 1):
                    text += f"Step {i}: {step}\n"
                text += "</think>\n\n"
            text += f"{example['answer']}<|im_end|>"
        else:
            # Generic format
            text = example.get('text', str(example))

        # Tokenize
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # Add labels (same as input_ids for causal LM)
        encoded['labels'] = encoded['input_ids'].copy()

        return encoded

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
    print("  Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Zen1 on Qwen2.5-32B")
    parser.add_argument("--dataset", default="data/reasoning.jsonl",
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
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print_banner("Zen1 Training - Qwen2.5-32B")
    print(f"\n📋 Configuration:")
    print(f"  Base Model: Qwen2.5-32B-Instruct (Latest)")
    print(f"  Parameters: 32 Billion")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA rank: 128")

    # Load model
    model, tokenizer = load_model_and_tokenizer()

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
        save_steps=100,
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
    print("  Fine-tuning Qwen2.5-32B with LoRA rank 128")
    print("  This is the most powerful Qwen model available")

    try:
        trainer.train()

        # Save model
        print(f"\n💾 Saving model to {output_dir}...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # Save training info
        info = {
            "base_model": "Qwen/Qwen2.5-32B-Instruct",
            "model_size": "32B parameters",
            "training_dataset": args.dataset,
            "epochs": args.epochs,
            "lora_rank": 128,
            "lora_alpha": 256,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "training_examples": len(train_dataset),
        }

        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print_banner("✅ Training Complete!")
        print(f"\nYour Zen1 model is ready!")
        print(f"  Base: Qwen2.5-32B (Latest and most powerful)")
        print(f"  LoRA Rank: 128")
        print(f"  Model saved to: {output_dir}")
        print(f"\nTo use the model:")
        print(f"  from peft import PeftModel")
        print(f"  model = PeftModel.from_pretrained(")
        print(f'      base_model, "{output_dir}"')
        print(f"  )")

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