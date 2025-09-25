#!/usr/bin/env python3
"""
Zen1-Thinking: Fine-tune for chain-of-thought reasoning
Simple and efficient local training
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm


def load_model_and_tokenizer():
    """Load Zen1 Thinking model"""
    # Using Qwen3-Omni-30B-A3B-Thinking as base
    model_name = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    print(f"Loading {model_name}...")

    # 4-bit quantization for large model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Add thinking tokens
    thinking_tokens = ["<think>", "</think>", "<step>"]
    tokenizer.add_special_tokens({"additional_special_tokens": thinking_tokens})
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def apply_lora(model):
    """Apply LoRA for efficient fine-tuning"""
    lora_config = LoraConfig(
        r=128,  # Rank
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_dataset(file_path, tokenizer):
    """Load reasoning dataset"""
    data = []

    # Load JSONL file
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)

            # Format for chain-of-thought
            prompt = f"Question: {item['question']}\n\n<think>\n"

            # Add reasoning steps
            for i, step in enumerate(item.get('reasoning', []), 1):
                prompt += f"<step> {i}: {step}\n"

            prompt += f"</think>\n\nAnswer: {item['answer']}"

            # Tokenize
            tokens = tokenizer(prompt, truncation=True, max_length=4096)
            data.append(tokens)

    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="Train Zen1-Thinking model")
    parser.add_argument("--dataset", default="data/reasoning.jsonl", help="Training data")
    parser.add_argument("--output", default="models/zen1-thinking", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)

    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    train_dataset = load_dataset(args.dataset, tokenizer)
    print(f"Loaded {len(train_dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {args.output}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)

    print("✓ Zen1-Thinking training complete!")


if __name__ == "__main__":
    main()