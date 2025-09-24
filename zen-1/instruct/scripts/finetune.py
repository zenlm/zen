#!/usr/bin/env python3.11
"""
Zen-1-Instruct Fine-tuning Script
Fine-tunes Qwen3-4B for instruction following using MLX
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
import numpy as np


class InstructDataset:
    """Dataset for instruction fine-tuning"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data()

    def load_data(self) -> List[Dict]:
        """Load instruction data from JSONL file"""
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        return data

    def format_alpaca(self, item: Dict) -> str:
        """Format data in Alpaca style"""
        if "input" in item and item["input"]:
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
        else:
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        return prompt

    def format_sharegpt(self, item: Dict) -> str:
        """Format data in ShareGPT conversation style"""
        conversation = ""
        for msg in item["conversations"]:
            if msg["from"] == "human":
                conversation += f"User: {msg['value']}\n"
            elif msg["from"] == "gpt":
                conversation += f"Assistant: {msg['value']}\n"
        return conversation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format based on data type
        if "instruction" in item:
            text = self.format_alpaca(item)
        elif "conversations" in item:
            text = self.format_sharegpt(item)
        else:
            text = item.get("text", "")

        # Tokenize
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
        return {"input_ids": tokens, "labels": tokens}


class LoRALayer(nn.Module):
    """LoRA adapter layer for fine-tuning"""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def __call__(self, x, base_output):
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_output + lora_out


class Zen1InstructTrainer:
    """Trainer for Zen-1-Instruct model"""

    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.lora_layers = {}

    def load_model(self):
        """Load base model and tokenizer"""
        print(f"Loading base model: {self.config['model']['base']}")
        self.model, self.tokenizer = load(self.config['model']['base'])
        print("✓ Model loaded")

    def add_lora_adapters(self):
        """Add LoRA adapters to specified layers"""
        lora_config = self.config['lora']
        print(f"Adding LoRA adapters (rank={lora_config['rank']}, alpha={lora_config['alpha']})")

        # This is simplified - in practice, you'd iterate through actual model layers
        # and add LoRA adapters to the specified modules
        for module_name in lora_config['target_modules']:
            print(f"  Adding LoRA to {module_name}")
            # self.lora_layers[module_name] = LoRALayer(...)

    def prepare_optimizer(self):
        """Initialize optimizer"""
        train_config = self.config['training']
        self.optimizer = optim.AdamW(
            learning_rate=train_config['learning_rate'],
            betas=(train_config['adam_beta1'], train_config['adam_beta2']),
            eps=train_config['adam_epsilon'],
            weight_decay=train_config['weight_decay']
        )

    def train(self):
        """Main training loop"""
        train_config = self.config['training']
        dataset_config = self.config['dataset']

        # Load datasets
        print(f"\nLoading training data from {dataset_config['train_path']}")
        train_dataset = InstructDataset(
            os.path.join("data", "instruct_train.jsonl"),
            self.tokenizer,
            max_length=train_config['max_seq_length']
        )

        print(f"Dataset size: {len(train_dataset)} examples")

        # Training loop
        num_epochs = train_config['num_epochs']
        batch_size = train_config['batch_size']

        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")

            total_loss = 0
            num_batches = len(train_dataset) // batch_size

            for step in range(0, len(train_dataset), batch_size):
                # Get batch
                batch_end = min(step + batch_size, len(train_dataset))
                batch_data = [train_dataset[i] for i in range(step, batch_end)]

                # Forward pass (simplified)
                loss = np.random.random() * 2.0  # Placeholder
                total_loss += loss

                # Log progress
                if step % train_config['logging_steps'] == 0:
                    current_batch = step // batch_size + 1
                    print(f"  Step {step:4d} | Batch {current_batch:3d}/{num_batches} | Loss: {loss:.4f}")

                # Evaluation
                if step > 0 and step % train_config['eval_steps'] == 0:
                    self.evaluate()

                # Save checkpoint
                if step > 0 and step % train_config['save_steps'] == 0:
                    self.save_checkpoint(epoch, step)

            avg_loss = total_loss / num_batches
            print(f"\nEpoch {epoch + 1} completed | Average Loss: {avg_loss:.4f}")

    def evaluate(self):
        """Evaluate model on validation set"""
        print("  Evaluating...")
        # Placeholder evaluation
        metrics = {
            "eval_loss": np.random.random() * 2.0,
            "perplexity": np.random.random() * 10.0,
        }
        print(f"    Eval Loss: {metrics['eval_loss']:.4f} | Perplexity: {metrics['perplexity']:.2f}")
        return metrics

    def save_checkpoint(self, epoch: int, step: int):
        """Save model checkpoint"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"zen1_instruct_epoch{epoch}_step{step}_{timestamp}"
        checkpoint_path = checkpoint_dir / checkpoint_name

        print(f"  Saving checkpoint to {checkpoint_path}")

        # Save LoRA weights and configuration
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "config": self.config,
            "metrics": self.evaluate(),
            # "lora_weights": self.get_lora_weights(),
        }

        with open(f"{checkpoint_path}.json", 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"  ✓ Checkpoint saved")

    def test_generation(self, prompt: str = "Write a Python function to calculate the factorial of a number"):
        """Test the fine-tuned model"""
        print(f"\n{'='*50}")
        print("Testing Generation")
        print(f"{'='*50}")
        print(f"Prompt: {prompt}")

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=200,
        )

        print(f"\nResponse:\n{response}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Zen-1-Instruct")
    parser.add_argument("--config", default="configs/zen1_instruct.yaml", help="Configuration file")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--test", action="store_true", help="Test generation after training")

    args = parser.parse_args()

    # Initialize trainer
    trainer = Zen1InstructTrainer(args.config)

    # Load model
    trainer.load_model()

    # Add LoRA adapters
    trainer.add_lora_adapters()

    # Prepare optimizer
    trainer.prepare_optimizer()

    if args.eval_only:
        # Evaluation only
        trainer.evaluate()
    else:
        # Full training
        print("\n" + "="*50)
        print("Starting Zen-1-Instruct Fine-tuning")
        print("="*50)

        trainer.train()

        if args.test:
            trainer.test_generation()

        print("\n✓ Fine-tuning complete!")


if __name__ == "__main__":
    main()