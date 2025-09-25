#!/usr/bin/env python3
"""Simple training script for v1.0.1 models using transformers"""

import json
import os
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

def prepare_dataset(jsonl_file):
    """Load and prepare training data"""
    examples = []
    with open(jsonl_file) as f:
        for line in f:
            data = json.loads(line)
            # Format as instruction-response pair
            text = f"### Instruction:\n{data['instruction']}\n\n### Response:\n{data['response']}"
            examples.append({"text": text})
    return Dataset.from_list(examples)

def train_model(model_name, base_model, output_dir):
    """Train a model with recursive improvements"""
    print(f"\n🚀 Training {model_name}...")

    # Load base model and tokenizer
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    dataset = prepare_dataset("training_data_v1.1.jsonl")

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="no",
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to=[]
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Training complete for {model_name}")
    return output_dir

def main():
    """Train all v1.0.1 models"""

    models = [
        {
            "name": "zen-nano-instruct-v1.0.1",
            "base": "zenlm/zen-nano-instruct",
            "output": "models/zen-nano-instruct-v1.0.1"
        },
        {
            "name": "zen-nano-thinking-v1.0.1",
            "base": "zenlm/zen-nano-thinking",
            "output": "models/zen-nano-thinking-v1.0.1"
        }
    ]

    # Check if we're actually going to train
    if not torch.cuda.is_available():
        print("⚠️  No GPU available. Training will be slow on CPU.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    # Train each model
    for model_config in models:
        try:
            train_model(
                model_config["name"],
                model_config["base"],
                model_config["output"]
            )
        except Exception as e:
            print(f"❌ Error training {model_config['name']}: {e}")
            print("Continuing with next model...")

    print("\n🎉 All training complete!")

if __name__ == "__main__":
    main()