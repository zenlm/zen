#!/usr/bin/env python3
"""
Qwen3-Omni Fine-tuning Script
Supports MoE architecture with LoRA adapters for multimodal training
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import soundfile as sf
import numpy as np
from tqdm import tqdm
import wandb

# Import Qwen3-Omni utilities
try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    print("Please install qwen-omni-utils: pip install qwen-omni-utils")
    sys.exit(1)


@dataclass
class Qwen3OmniTrainingArguments(TrainingArguments):
    """Extended training arguments for Qwen3-Omni"""

    # Multimodal settings
    use_audio_in_video: bool = field(default=True)
    generate_audio: bool = field(default=False)

    # MoE specific
    moe_load_balancing_loss_coef: float = field(default=0.01)
    expert_capacity: float = field(default=1.25)

    # LoRA settings
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.1)

    # Loss weights
    text_loss_weight: float = field(default=1.0)
    audio_loss_weight: float = field(default=0.9)
    vision_loss_weight: float = field(default=0.85)


class MultimodalDataset(Dataset):
    """Dataset for multimodal fine-tuning"""

    def __init__(self, data_path: str, processor, config: Dict):
        self.data_path = data_path
        self.processor = processor
        self.config = config
        self.data = self.load_data()

    def load_data(self) -> List[Dict]:
        """Load multimodal training data"""
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract conversations
        conversations = item.get("conversations", [])

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False
        )

        # Process multimodal data
        audios, images, videos = process_mm_info(
            conversations,
            use_audio_in_video=self.config.get("use_audio_in_video", True)
        )

        # Tokenize and prepare inputs
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.get("max_length", 32768),
            truncation=True,
            use_audio_in_video=self.config.get("use_audio_in_video", True)
        )

        # Handle labels for training
        inputs["labels"] = inputs["input_ids"].clone()

        return {k: v.squeeze(0) for k, v in inputs.items()}


class MoELoRAModule(nn.Module):
    """Custom LoRA module for MoE architecture"""

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 64, alpha: float = 128,
                 num_experts: int = 8):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.num_experts = num_experts

        # LoRA matrices for each expert
        self.lora_A = nn.ModuleList([
            nn.Linear(in_features, rank, bias=False)
            for _ in range(num_experts)
        ])
        self.lora_B = nn.ModuleList([
            nn.Linear(rank, out_features, bias=False)
            for _ in range(num_experts)
        ])

        # Initialize weights
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[i].weight)
            nn.init.zeros_(self.lora_B[i].weight)

    def forward(self, x, expert_weights):
        """Forward pass with expert routing"""
        batch_size = x.shape[0]
        output = torch.zeros_like(x[:, :, :self.lora_B[0].out_features])

        for i in range(self.num_experts):
            # Apply LoRA for each expert
            expert_out = self.lora_B[i](self.lora_A[i](x)) * self.scaling

            # Weight by expert routing
            if expert_weights is not None:
                expert_out = expert_out * expert_weights[:, i:i+1, None]

            output += expert_out

        return output


class Qwen3OmniTrainer(Trainer):
    """Custom trainer for Qwen3-Omni with MoE support"""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute multimodal loss with modality weighting"""

        # Forward pass
        outputs = model(**inputs)

        # Get base loss
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        # Add MoE load balancing loss if available
        if hasattr(outputs, 'aux_loss'):
            aux_loss = outputs.aux_loss * self.args.moe_load_balancing_loss_coef
            loss = loss + aux_loss

        # Apply modality-specific loss weights
        if hasattr(self.args, 'text_loss_weight'):
            # This would require tracking which tokens correspond to which modality
            # Simplified version - just scale overall loss
            loss = loss * self.args.text_loss_weight

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step for multimodal evaluation"""
        with torch.no_grad():
            outputs = model(**inputs)
            loss = self.compute_loss(model, inputs)

            # Generate predictions if needed
            if not prediction_loss_only:
                # Generate text
                text_ids, audio = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.args.use_audio_in_video
                )

                return loss, text_ids, inputs.get("labels")

            return loss, None, None


def setup_model_and_tokenizer(config: Dict):
    """Initialize model with quantization and LoRA"""

    model_name = config['model']['name']

    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get('load_in_4bit', True),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load model
    print(f"Loading model: {model_name}")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config if config.get('use_quantization', True) else None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    # Load processor
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)

    # Prepare model for k-bit training
    if config.get('use_quantization', True):
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config['lora']['rank'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        bias=config['lora'].get('bias', 'none'),
        task_type=TaskType.MULTIMODAL,
        target_modules=config['lora']['target_modules']
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-Omni")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--deepspeed", help="Path to DeepSpeed config")
    parser.add_argument("--resume", help="Resume from checkpoint")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    if config['training'].get('report_to', []):
        wandb.init(
            project="qwen3-omni-finetuning",
            name=config.get('run_name', 'qwen3-omni'),
            config=config
        )

    # Setup model and processor
    model, processor = setup_model_and_tokenizer(config)

    # Load datasets
    print("Loading datasets...")
    train_dataset = MultimodalDataset(
        config['dataset']['train_path'],
        processor,
        config['dataset']
    )

    eval_dataset = None
    if config['dataset'].get('val_path'):
        eval_dataset = MultimodalDataset(
            config['dataset']['val_path'],
            processor,
            config['dataset']
        )

    # Training arguments
    training_args = Qwen3OmniTrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],

        # Learning rate
        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],

        # Optimization
        optim=config['training']['optim'],
        adam_beta1=config['training']['adam_beta1'],
        adam_beta2=config['training']['adam_beta2'],
        adam_epsilon=config['training']['adam_epsilon'],
        max_grad_norm=config['training']['max_grad_norm'],

        # Memory
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        bf16=config['training']['bf16'],
        tf32=config['training'].get('tf32', True),

        # Evaluation
        evaluation_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],

        # Logging
        logging_steps=config['training']['logging_steps'],
        logging_first_step=config['training']['logging_first_step'],
        report_to=config['training']['report_to'],

        # Custom args
        use_audio_in_video=config['dataset']['modalities']['use_audio_in_video'],
        lora_rank=config['lora']['rank'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],

        # DeepSpeed
        deepspeed=args.deepspeed,

        # Resume
        resume_from_checkpoint=args.resume,
    )

    # Initialize trainer
    trainer = Qwen3OmniTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
    )

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # Save final model
    print("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)

    # Save LoRA weights separately
    lora_path = Path(training_args.output_dir) / "lora_adapters"
    model.save_pretrained(lora_path)

    print(f"Training complete! Model saved to {training_args.output_dir}")

    # Cleanup
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()