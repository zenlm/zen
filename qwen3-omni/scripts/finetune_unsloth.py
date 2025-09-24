#!/usr/bin/env python3
"""
Qwen3-Omni Fine-tuning with Unsloth
Ultra-efficient training with 2-5x speedup and 80% less memory
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import wandb


# Unsloth optimized models mapping
UNSLOTH_MODELS = {
    "qwen3-omni-30b": "unsloth/Qwen3-Omni-30B-A3B-bnb-4bit",
    "qwen3-omni-instruct": "unsloth/Qwen3-Omni-30B-A3B-Instruct-bnb-4bit",
    "qwen3-omni-thinking": "unsloth/Qwen3-Omni-30B-A3B-Thinking-bnb-4bit",
}


@dataclass
class UnslothTrainingConfig:
    """Configuration for Unsloth fine-tuning"""

    # Model settings
    model_name: str = "qwen3-omni-instruct"
    max_seq_length: int = 32768
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = True

    # LoRA settings
    lora_r: int = 128  # Rank - higher for complex tasks
    lora_alpha: int = 256  # Alpha scaling
    lora_dropout: float = 0.05
    lora_bias: str = "none"

    # Target modules for Qwen3-Omni MoE
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate", "up_proj", "down_proj",
        "experts.*.wi", "experts.*.wo",
        "thinker.*", "talker.*",
        "embed_tokens", "lm_head"
    ])

    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3

    # Optimization
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine"
    seed: int = 42

    # Data settings
    dataset_path: str = "data/multimodal/train.jsonl"
    val_split: float = 0.1
    packing: bool = True  # Efficient sequence packing

    # Output
    output_dir: str = "checkpoints/unsloth"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


class Qwen3OmniUnslothTrainer:
    """Unsloth-optimized trainer for Qwen3-Omni"""

    def __init__(self, config: UnslothTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def setup_model(self):
        """Initialize model with Unsloth optimizations"""

        # Get Unsloth model name or use custom
        model_name = UNSLOTH_MODELS.get(
            self.config.model_name,
            self.config.model_name
        )

        print(f"Loading model with Unsloth: {model_name}")

        # Load with Unsloth's FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,

            # Unsloth specific optimizations
            rope_scaling={"type": "linear", "factor": 2.0},  # RoPE scaling
            use_gradient_checkpointing="unsloth",  # Efficient checkpointing
            random_state=self.config.seed,
            use_rslora=True,  # Rank-stabilized LoRA
            loftq_config={},  # LoftQ quantization
        )

        # Apply LoRA with Unsloth optimizations
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            target_modules=self.config.target_modules,

            # Unsloth enhancements
            use_gradient_checkpointing="unsloth",
            modules_to_save=["embed_tokens", "lm_head"],
            use_rslora=True,  # Rank-stabilized LoRA for better training
            use_dora=False,  # Weight-decomposed LoRA (optional)
            loftq_config={},  # Quantization-aware fine-tuning
        )

        # Setup chat template for Qwen3
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen-3",
            mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"}
        )

        print(f"Model loaded with {self.count_parameters()} trainable parameters")

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def prepare_multimodal_dataset(self, dataset_path: str) -> Dataset:
        """Prepare multimodal dataset for training"""

        def formatting_func(examples):
            """Format multimodal conversations"""
            texts = []

            for conversation in examples["conversations"]:
                # Format with chat template
                formatted = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(formatted)

            return {"text": texts}

        # Load dataset
        if dataset_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            dataset = load_dataset(dataset_path, split='train')

        # Format for training
        dataset = dataset.map(
            formatting_func,
            batched=True,
            remove_columns=dataset.column_names
        )

        return dataset

    def train(self):
        """Run Unsloth-optimized training"""

        # Prepare dataset
        dataset = self.prepare_multimodal_dataset(self.config.dataset_path)

        # Split train/val
        if self.config.val_split > 0:
            split = dataset.train_test_split(test_size=self.config.val_split, seed=self.config.seed)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = dataset
            eval_dataset = None

        # Training arguments optimized for Unsloth
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,

            # Batch settings
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,

            # Learning rate
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,

            # Optimization
            optim=self.config.optim,
            max_grad_norm=self.config.max_grad_norm,

            # Memory optimization
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),

            # Logging
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if eval_dataset else "no",

            # Misc
            seed=self.config.seed,
            report_to="wandb",
            run_name=f"qwen3-omni-unsloth-{self.config.model_name}",
        )

        # Initialize trainer with Unsloth optimizations
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,

            # SFT specific
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=self.config.packing,  # Efficient sequence packing

            # Data collator for efficient batching
            data_collator=DataCollatorForCompletionOnlyLM(
                response_template="<|assistant|>",
                instruction_template="<|user|>",
                tokenizer=self.tokenizer,
                mlm=False,
            ),
        )

        # Start training
        print("Starting Unsloth-optimized training...")
        trainer.train()

        # Save model
        print("Saving model...")
        trainer.save_model()

        # Save to 16bit for deployment
        self.save_16bit()

    def save_16bit(self):
        """Save model in 16bit for efficient inference"""
        print("Saving 16bit model for deployment...")

        # Merge LoRA weights and save
        self.model.save_pretrained_merged(
            f"{self.config.output_dir}_16bit",
            self.tokenizer,
            save_method="merged_16bit",
            max_shard_size="10GB"
        )

    def save_gguf(self, quantization: str = "q4_k_m"):
        """Export to GGUF format for llama.cpp"""
        print(f"Exporting to GGUF format ({quantization})...")

        self.model.save_pretrained_gguf(
            f"{self.config.output_dir}_gguf",
            self.tokenizer,
            quantization_method=quantization
        )

    def push_to_hub(self, repo_name: str):
        """Push model to Hugging Face Hub"""
        print(f"Pushing to Hub: {repo_name}")

        self.model.push_to_hub_merged(
            repo_name,
            self.tokenizer,
            save_method="merged_16bit",
            token=os.getenv("HF_TOKEN")
        )


def create_multimodal_prompt(sample: Dict) -> str:
    """Create formatted prompt for multimodal input"""

    prompt_parts = []

    # System prompt for multimodal
    system = """You are Qwen3-Omni, a multimodal AI assistant capable of understanding:
- Text in 119 languages
- Audio in 19 languages
- Images and videos
- Complex reasoning across modalities

Respond naturally and concisely."""

    prompt_parts.append(f"<|system|>\n{system}")

    # Add user content
    user_content = []
    for item in sample.get("content", []):
        if item["type"] == "text":
            user_content.append(item["text"])
        elif item["type"] == "audio":
            user_content.append(f"[Audio: {item['audio']}]")
        elif item["type"] == "image":
            user_content.append(f"[Image: {item['image']}]")
        elif item["type"] == "video":
            user_content.append(f"[Video: {item['video']}]")

    prompt_parts.append(f"<|user|>\n{' '.join(user_content)}")

    # Add assistant response if training
    if "response" in sample:
        prompt_parts.append(f"<|assistant|>\n{sample['response']}")

    return "\n".join(prompt_parts)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-Omni with Unsloth")
    parser.add_argument("--config", help="Config YAML file")
    parser.add_argument("--model", default="qwen3-omni-instruct", help="Model variant")
    parser.add_argument("--dataset", default="data/multimodal/train.jsonl", help="Dataset path")
    parser.add_argument("--output", default="checkpoints/unsloth", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank")
    parser.add_argument("--save_16bit", action="store_true", help="Save 16bit model")
    parser.add_argument("--save_gguf", action="store_true", help="Export to GGUF")
    parser.add_argument("--push_to_hub", help="Push to HF Hub repo")

    args = parser.parse_args()

    # Load config from YAML or create from args
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = UnslothTrainingConfig(**config_dict)
    else:
        config = UnslothTrainingConfig(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            lora_r=args.lora_r
        )

    # Initialize trainer
    trainer = Qwen3OmniUnslothTrainer(config)

    # Setup model
    trainer.setup_model()

    # Train
    trainer.train()

    # Save in different formats
    if args.save_16bit:
        trainer.save_16bit()

    if args.save_gguf:
        trainer.save_gguf()

    if args.push_to_hub:
        trainer.push_to_hub(args.push_to_hub)

    print("Training complete!")


if __name__ == "__main__":
    main()