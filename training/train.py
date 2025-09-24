#!/usr/bin/env python3
"""
Zen Model Training Pipeline
Main training script with unified interface for all model variants
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, Dataset

from utils.data_preparation import prepare_datasets
from utils.model_loader import load_base_model
from utils.training_utils import setup_training, save_checkpoint
from configs.model_configs import get_model_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ZenTrainingConfig:
    """Unified configuration for Zen model training"""
    model_variant: str = "zen-nano"
    base_model: str = "Qwen/Qwen3-4B"
    training_stage: str = "instruct"  # base, instruct, thinking
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # QLoRA parameters
    use_qlora: bool = False
    quantization_bits: int = 4
    
    # BitDelta parameters
    use_bitdelta: bool = False
    bitdelta_ratio: float = 0.1
    
    # Data parameters
    dataset_path: str = "./data"
    train_split: float = 0.9
    validation_split: float = 0.05
    test_split: float = 0.05
    
    # Output parameters
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    save_steps: int = 500
    eval_steps: int = 100
    
    # Hardware optimization
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    
    # Advanced features
    use_flash_attention: bool = True
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "ZenTrainingConfig":
        """Load configuration from YAML file"""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class ZenTrainer:
    """Main trainer class for Zen models"""
    
    def __init__(self, config: ZenTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup_model(self):
        """Initialize model with LoRA/QLoRA if configured"""
        logger.info(f"Loading base model: {self.config.base_model}")
        
        # Setup quantization config if using QLoRA
        bnb_config = None
        if self.config.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=self.config.use_flash_attention
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup LoRA if configured
        if self.config.use_lora or self.config.use_qlora:
            if self.config.use_qlora:
                self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
    def load_datasets(self):
        """Load and prepare training datasets"""
        logger.info("Loading datasets...")
        
        # Load datasets based on training stage
        if self.config.training_stage == "base":
            train_dataset = self._load_base_dataset()
        elif self.config.training_stage == "instruct":
            train_dataset = self._load_instruct_dataset()
        elif self.config.training_stage == "thinking":
            train_dataset = self._load_thinking_dataset()
        else:
            raise ValueError(f"Unknown training stage: {self.config.training_stage}")
        
        # Split into train/val/test
        train_size = int(len(train_dataset) * self.config.train_split)
        val_size = int(len(train_dataset) * self.config.validation_split)
        
        train_dataset = train_dataset.select(range(train_size))
        val_dataset = train_dataset.select(range(train_size, train_size + val_size))
        
        return train_dataset, val_dataset
    
    def _load_base_dataset(self):
        """Load dataset for base model training"""
        # Load from local files or HuggingFace
        dataset = load_dataset(
            "json",
            data_files=f"{self.config.dataset_path}/base_training.jsonl",
            split="train"
        )
        return self._tokenize_dataset(dataset)
    
    def _load_instruct_dataset(self):
        """Load dataset for instruction tuning"""
        dataset = load_dataset(
            "json",
            data_files=f"{self.config.dataset_path}/instruct_training.jsonl",
            split="train"
        )
        return self._tokenize_dataset(dataset)
    
    def _load_thinking_dataset(self):
        """Load dataset for thinking/CoT training"""
        dataset = load_dataset(
            "json",
            data_files=f"{self.config.dataset_path}/thinking_training.jsonl",
            split="train"
        )
        return self._tokenize_dataset(dataset)
    
    def _tokenize_dataset(self, dataset):
        """Tokenize dataset"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_length
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        return tokenized
    
    def setup_trainer(self, train_dataset, val_dataset):
        """Setup HuggingFace Trainer"""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_dir=self.config.logging_dir,
            logging_steps=10,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim=self.config.optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            deepspeed=self.config.deepspeed_config,
            report_to=["tensorboard"],
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.config.model_variant}")
        
        # Setup model
        self.setup_model()
        
        # Load datasets
        train_dataset, val_dataset = self.load_datasets()
        
        # Setup trainer
        self.setup_trainer(train_dataset, val_dataset)
        
        # Train
        self.trainer.train()
        
        # Save final model
        self.save_model()
        
        logger.info("Training completed!")
        
    def save_model(self):
        """Save trained model and configuration"""
        output_path = Path(self.config.output_dir) / self.config.model_variant
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.trainer.save_model(str(output_path))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_path))
        
        # Save training config
        with open(output_path / "training_config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        logger.info(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Zen models")
    parser.add_argument(
        "--model",
        type=str,
        default="zen-nano",
        choices=["zen-nano", "zen-omni", "zen-coder", "zen-next"],
        help="Model variant to train"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="instruct",
        choices=["base", "instruct", "thinking"],
        help="Training stage"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for efficient training"
    )
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        help="Use QLoRA for memory-efficient training"
    )
    parser.add_argument(
        "--use-bitdelta",
        action="store_true",
        help="Use BitDelta for personalization"
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
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = ZenTrainingConfig.from_yaml(args.config)
    else:
        # Get model-specific config
        model_config = get_model_config(args.model)
        
        config = ZenTrainingConfig(
            model_variant=args.model,
            base_model=model_config["base_model"],
            training_stage=args.stage,
            use_lora=args.use_lora,
            use_qlora=args.use_qlora,
            use_bitdelta=args.use_bitdelta,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            **model_config.get("training_params", {})
        )
    
    # Initialize and run trainer
    trainer = ZenTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()