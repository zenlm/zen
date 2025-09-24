#!/usr/bin/env python3
"""
Qwen3-Omni Training on HuggingFace with GSPO
Group Symmetry Preserving Optimization for preference alignment
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from accelerate import Accelerator
import wandb
from huggingface_hub import HfApi, create_repo, upload_folder


@dataclass
class GSPOConfig:
    """Configuration for Group Symmetry Preserving Optimization"""

    # Model settings
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    model_revision: str = "main"
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    use_flash_attention: bool = True

    # GSPO specific parameters
    beta: float = 0.1  # KL regularization coefficient
    group_size: int = 4  # Group size for symmetry preservation
    symmetry_weight: float = 0.5  # Weight for symmetry loss
    reference_free: bool = False  # Use reference-free variant
    label_smoothing: float = 0.0  # Label smoothing for robustness

    # Loss weights for multimodal
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "text": 1.0,
        "audio": 0.9,
        "vision": 0.85,
        "cross_modal": 1.1
    })

    # LoRA configuration
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate", "up_proj", "down_proj",
        "experts.*.wi", "experts.*.wo"
    ])

    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # HuggingFace settings
    hub_model_id: str = "your-username/qwen3-omni-gspo"
    hub_private: bool = False
    push_to_hub: bool = True
    hub_strategy: str = "every_save"

    # Dataset settings
    dataset_name: str = "your-username/qwen3-omni-preferences"
    dataset_split: str = "train"
    max_seq_length: int = 8192
    max_prompt_length: int = 4096

    # Optimization
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    report_to: str = "wandb"
    run_name: str = "qwen3-omni-gspo"


class GSPOTrainer(DPOTrainer):
    """
    GSPO Trainer extending DPO with group symmetry preservation
    """

    def __init__(self, *args, gspo_config: GSPOConfig = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gspo_config = gspo_config or GSPOConfig()

    def compute_gspo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute GSPO loss with group symmetry preservation
        """
        # Standard DPO loss
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        # DPO loss
        dpo_loss = -F.logsigmoid(self.gspo_config.beta * logits).mean()

        # Group Symmetry Loss
        batch_size = policy_chosen_logps.shape[0]
        group_size = min(self.gspo_config.group_size, batch_size)

        if group_size > 1:
            # Reshape for groups
            num_groups = batch_size // group_size
            if num_groups > 0:
                grouped_chosen = policy_chosen_logps[:num_groups * group_size].reshape(num_groups, group_size)
                grouped_rejected = policy_rejected_logps[:num_groups * group_size].reshape(num_groups, group_size)

                # Compute group symmetry (variance within groups should be small)
                group_var_chosen = grouped_chosen.var(dim=1).mean()
                group_var_rejected = grouped_rejected.var(dim=1).mean()
                symmetry_loss = (group_var_chosen + group_var_rejected) * self.gspo_config.symmetry_weight
            else:
                symmetry_loss = torch.tensor(0.0, device=policy_chosen_logps.device)
        else:
            symmetry_loss = torch.tensor(0.0, device=policy_chosen_logps.device)

        # Total loss
        loss = dpo_loss + symmetry_loss

        # Compute rewards for logging
        chosen_rewards = self.gspo_config.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.gspo_config.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to use GSPO
        """
        # Get logprobs from model
        policy_chosen_logps = self.get_batch_logps(
            model,
            inputs["input_ids_chosen"],
            inputs["attention_mask_chosen"],
            inputs["labels_chosen"]
        )

        policy_rejected_logps = self.get_batch_logps(
            model,
            inputs["input_ids_rejected"],
            inputs["attention_mask_rejected"],
            inputs["labels_rejected"]
        )

        # Get reference logprobs
        with torch.no_grad():
            if self.ref_model is not None:
                reference_chosen_logps = self.get_batch_logps(
                    self.ref_model,
                    inputs["input_ids_chosen"],
                    inputs["attention_mask_chosen"],
                    inputs["labels_chosen"]
                )
                reference_rejected_logps = self.get_batch_logps(
                    self.ref_model,
                    inputs["input_ids_rejected"],
                    inputs["attention_mask_rejected"],
                    inputs["labels_rejected"]
                )
            else:
                # Reference-free variant
                reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
                reference_rejected_logps = torch.zeros_like(policy_rejected_logps)

        # Compute GSPO loss
        loss, chosen_rewards, rejected_rewards = self.compute_gspo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )

        # Add multimodal modality weighting if available
        if "modality_mask" in inputs:
            modality_weights = self._get_modality_weights(inputs["modality_mask"])
            loss = loss * modality_weights.mean()

        if return_outputs:
            return loss, {
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
            }
        return loss

    def _get_modality_weights(self, modality_mask):
        """Apply modality-specific loss weights"""
        weights = torch.ones_like(modality_mask, dtype=torch.float32)
        for modality, weight in self.gspo_config.modality_weights.items():
            if modality in ["text", "audio", "vision"]:
                # This would require tracking which tokens belong to which modality
                # Simplified implementation
                weights = weights * weight
        return weights


def create_multimodal_dataset(dataset_name: str, tokenizer, max_length: int = 8192):
    """
    Create preference dataset for multimodal GSPO training
    """

    # Load dataset from HuggingFace Hub
    dataset = load_dataset(dataset_name, split="train")

    def preprocess_function(examples):
        """Preprocess multimodal preference pairs"""

        chosen_messages = []
        rejected_messages = []

        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            # Format chosen response
            chosen_text = tokenizer.apply_chat_template(
                chosen["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            chosen_messages.append(chosen_text)

            # Format rejected response
            rejected_text = tokenizer.apply_chat_template(
                rejected["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            rejected_messages.append(rejected_text)

        # Tokenize
        tokenized_chosen = tokenizer(
            chosen_messages,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        tokenized_rejected = tokenizer(
            rejected_messages,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }

    # Process dataset
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return dataset


def setup_model_for_training(config: GSPOConfig):
    """
    Setup model with LoRA and quantization for HF training
    """

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        quantization_config=bnb_config if config.load_in_4bit else None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attention_2=config.use_flash_attention,
    )

    # Prepare for k-bit training
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)

    # Add LoRA
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer, peft_config


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-Omni with GSPO on HuggingFace")
    parser.add_argument("--model_name", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--dataset_name", default="your-username/qwen3-omni-preferences")
    parser.add_argument("--hub_model_id", default="your-username/qwen3-omni-gspo")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="GSPO beta parameter")
    parser.add_argument("--group_size", type=int, default=4, help="Group size for symmetry")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    # Initialize config
    config = GSPOConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        hub_model_id=args.hub_model_id,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        group_size=args.group_size,
        push_to_hub=args.push_to_hub,
        report_to="wandb" if args.use_wandb else "none"
    )

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="qwen3-omni-gspo",
            name=config.run_name,
            config=config.__dict__
        )

    # Setup model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer, peft_config = setup_model_for_training(config)

    # Load dataset
    print("Loading preference dataset...")
    train_dataset = create_multimodal_dataset(
        config.dataset_name,
        tokenizer,
        config.max_seq_length
    )

    # Training arguments
    training_args = DPOConfig(
        output_dir=f"./checkpoints/{config.run_name}",
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,

        # Optimization
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=True,
        tf32=True,

        # Logging
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        report_to=config.report_to,
        run_name=config.run_name,

        # HuggingFace Hub
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_private_repo=config.hub_private,
        hub_strategy=config.hub_strategy,

        # DPO specific
        beta=config.beta,
        max_prompt_length=config.max_prompt_length,
        max_length=config.max_seq_length,
    )

    # Initialize GSPO trainer
    trainer = GSPOTrainer(
        model=model,
        ref_model=None,  # Will create automatically
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,  # Add if you have eval set
        gspo_config=config,
    )

    # Train
    print("Starting GSPO training...")
    trainer.train()

    # Save final model
    print("Saving final model...")
    trainer.save_model()

    # Push to hub
    if config.push_to_hub:
        print(f"Pushing model to HuggingFace Hub: {config.hub_model_id}")
        trainer.push_to_hub()

    print("Training complete!")


if __name__ == "__main__":
    main()