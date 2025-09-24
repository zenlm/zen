#!/usr/bin/env python3
"""
Zen1 Training for Qwen3-Omni models
Handles the special MoE architecture
"""

import os
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings("ignore")


def load_model_and_tokenizer(model_type="thinking"):
    """Load Qwen3-Omni model with proper configuration"""

    # Model selection
    models = {
        "thinking": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "captioner": "Qwen/Qwen3-Omni-30B-A3B-Captioner"
    }

    model_name = models.get(model_type, models["thinking"])
    print(f"Loading {model_name}...")

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try loading with trust_remote_code
    try:
        # Load model with FP16 for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "20GiB", "cpu": "30GiB"}
        )
        print(f"✓ Loaded {model_name} successfully")
    except Exception as e:
        print(f"Failed to load Qwen3-Omni model: {e}")
        print("Falling back to Qwen2.5 for testing...")

        # Fallback to Qwen2.5
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def apply_lora(model):
    """Apply LoRA configuration optimized for MoE models"""

    # Check if model is MoE
    is_moe = hasattr(model.config, 'num_experts') or 'moe' in model.config.model_type.lower()

    if is_moe:
        print("Detected MoE architecture, adapting LoRA configuration...")
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "router",  # MoE router
        ]
    else:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    lora_config = LoraConfig(
        r=128,  # Rank
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def load_dataset(file_path, tokenizer, dataset_type="reasoning"):
    """Load and format dataset"""
    data = []

    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)

            if dataset_type == "reasoning":
                # Format for chain-of-thought
                prompt = f"Question: {item['question']}\n\n<think>\n"

                # Add reasoning steps if available
                if 'reasoning' in item:
                    for i, step in enumerate(item['reasoning'], 1):
                        prompt += f"Step {i}: {step}\n"

                prompt += f"</think>\n\nAnswer: {item['answer']}"
            else:
                # Format for conversation
                messages = item.get('messages', [])
                if messages:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                else:
                    prompt = f"User: {item.get('user', '')}\nAssistant: {item.get('assistant', '')}"

            # Tokenize
            tokens = tokenizer(
                prompt,
                truncation=True,
                max_length=2048,  # Reduced for memory
                padding="max_length",
                return_tensors="pt"
            )

            # Convert tensors to lists for Dataset
            tokens = {k: v.squeeze().tolist() for k, v in tokens.items()}
            data.append(tokens)

    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="Train Zen1 models on Qwen3-Omni")
    parser.add_argument("--model_type", default="thinking",
                       choices=["thinking", "instruct", "captioner"],
                       help="Model variant to train")
    parser.add_argument("--dataset", default="data/reasoning.jsonl",
                       help="Training dataset")
    parser.add_argument("--dataset_type", default="reasoning",
                       choices=["reasoning", "conversation"],
                       help="Dataset format type")
    parser.add_argument("--output", default="models/zen1",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    args = parser.parse_args()

    # Set output path based on model type
    output_dir = f"{args.output}-{args.model_type}"

    print(f"🚀 Training Zen1-{args.model_type.capitalize()}")
    print("=" * 50)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_type)

    # Apply LoRA
    model = apply_lora(model)

    # Load dataset
    print(f"\n📚 Loading dataset from {args.dataset}")
    train_dataset = load_dataset(args.dataset, tokenizer, args.dataset_type)
    print(f"✓ Loaded {len(train_dataset)} examples")

    # Training arguments optimized for memory
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        gradient_checkpointing=True,
        fp16=True,  # Use FP16 for memory
        optim="adamw_8bit",  # 8-bit optimizer
        report_to="none",
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print(f"\n🎯 Starting training...")
    print(f"   Model: Zen1-{args.model_type.capitalize()}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")

    try:
        trainer.train()

        # Save model
        print(f"\n💾 Saving model to {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        print(f"\n✅ Zen1-{args.model_type.capitalize()} training complete!")
        print(f"   Model saved to: {output_dir}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())