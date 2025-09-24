#!/usr/bin/env python3
"""
Zen1-Omni Training on HuggingFace with GSPO
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
class Zen1GSPOConfig:
    """Configuration for Zen1-Omni GSPO Training"""

    # Model settings
    model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    model_revision: str = "main"
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    use_flash_attention: bool = True

    # Zen1-Omni specific
    model_variant: str = "zen1-omni-instruct"  # instruct, thinking, captioner
    enable_multimodal: bool = True
    enable_speech_output: bool = True

    # GSPO parameters
    beta: float = 0.1  # KL regularization
    group_size: int = 4  # Group symmetry size
    symmetry_weight: float = 0.5  # Symmetry loss weight
    reference_free: bool = False
    label_smoothing: float = 0.0

    # Multimodal loss weights
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "text": 1.0,
        "audio": 0.9,
        "vision": 0.85,
        "cross_modal": 1.1,
        "speech_generation": 0.95
    })

    # LoRA configuration for Zen1
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate", "up_proj", "down_proj",
        "experts.*.wi", "experts.*.wo",
        "thinker.*", "talker.*"  # Zen1-Omni specific
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

    # HuggingFace Hub settings - Zen1 branding
    hub_model_id: str = "zen-ai/zen1-omni-30b-gspo"
    hub_org: str = "zen-ai"
    hub_private: bool = False
    push_to_hub: bool = True
    hub_strategy: str = "every_save"

    # Dataset settings
    dataset_name: str = "zen-ai/zen1-omni-preferences"
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
    run_name: str = "zen1-omni-gspo"

    # Zen1 special features
    enable_chain_of_thought: bool = True
    thinking_tokens: Dict[str, str] = field(default_factory=lambda: {
        "start": "<think>",
        "end": "</think>",
        "step": "<step>"
    })


class Zen1OmniGSPOTrainer(DPOTrainer):
    """
    Zen1-Omni GSPO Trainer with multimodal support
    """

    def __init__(self, *args, zen1_config: Zen1GSPOConfig = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.zen1_config = zen1_config or Zen1GSPOConfig()
        self.modality_tracker = {}

    def compute_zen1_gspo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Compute Zen1-Omni GSPO loss with multimodal weighting
        """
        # Standard DPO component
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        # DPO loss
        dpo_loss = -F.logsigmoid(self.zen1_config.beta * logits)

        # Apply modality weights if available
        if modality_masks:
            modality_weights = self._compute_modality_weights(modality_masks)
            dpo_loss = dpo_loss * modality_weights

        dpo_loss = dpo_loss.mean()

        # Group Symmetry Loss (Zen1 enhancement)
        batch_size = policy_chosen_logps.shape[0]
        group_size = min(self.zen1_config.group_size, batch_size)

        symmetry_loss = torch.tensor(0.0, device=policy_chosen_logps.device)
        if group_size > 1:
            num_groups = batch_size // group_size
            if num_groups > 0:
                # Reshape for groups
                grouped_chosen = policy_chosen_logps[:num_groups * group_size].reshape(num_groups, group_size)
                grouped_rejected = policy_rejected_logps[:num_groups * group_size].reshape(num_groups, group_size)

                # Compute intra-group consistency
                group_var_chosen = grouped_chosen.var(dim=1).mean()
                group_var_rejected = grouped_rejected.var(dim=1).mean()

                # Inter-group diversity (Zen1 innovation)
                group_mean_chosen = grouped_chosen.mean(dim=1)
                group_mean_rejected = grouped_rejected.mean(dim=1)
                inter_group_diversity = -group_mean_chosen.var() - group_mean_rejected.var()

                # Combined symmetry loss
                symmetry_loss = (
                    (group_var_chosen + group_var_rejected) * self.zen1_config.symmetry_weight
                    + inter_group_diversity * 0.1  # Encourage diversity
                )

        # Total loss
        loss = dpo_loss + symmetry_loss

        # Compute rewards for logging
        chosen_rewards = self.zen1_config.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.zen1_config.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        # Zen1 metrics
        metrics = {
            "dpo_loss": dpo_loss.item(),
            "symmetry_loss": symmetry_loss.item(),
            "chosen_rewards_mean": chosen_rewards.mean().item(),
            "rejected_rewards_mean": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item()
        }

        return loss, chosen_rewards, rejected_rewards, metrics

    def _compute_modality_weights(self, modality_masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply Zen1-Omni modality-specific weights"""
        combined_weights = torch.ones_like(next(iter(modality_masks.values())))

        for modality, mask in modality_masks.items():
            if modality in self.zen1_config.modality_weights:
                weight = self.zen1_config.modality_weights[modality]
                combined_weights = combined_weights * (1 - mask) + mask * weight

        return combined_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss for Zen1-Omni GSPO
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
                # Reference-free variant for Zen1
                reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
                reference_rejected_logps = torch.zeros_like(policy_rejected_logps)

        # Extract modality masks if present
        modality_masks = {}
        for key in ["text_mask", "audio_mask", "vision_mask", "speech_mask"]:
            if key in inputs:
                modality_masks[key.replace("_mask", "")] = inputs[key]

        # Compute Zen1 GSPO loss
        loss, chosen_rewards, rejected_rewards, metrics = self.compute_zen1_gspo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            modality_masks
        )

        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            for key, value in metrics.items():
                self.log({f"zen1/{key}": value})

        if return_outputs:
            return loss, {
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
                **metrics
            }
        return loss


def create_zen1_dataset(dataset_name: str, tokenizer, config: Zen1GSPOConfig):
    """
    Create Zen1-Omni preference dataset with multimodal support
    """

    # Load from HuggingFace Hub
    dataset = load_dataset(dataset_name, split=config.dataset_split)

    def preprocess_zen1_sample(examples):
        """Preprocess Zen1-Omni multimodal preference pairs"""

        chosen_messages = []
        rejected_messages = []
        modality_info = []

        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            # Add Zen1 system prompt
            system_prompt = {
                "role": "system",
                "content": "You are Zen1-Omni, an advanced multimodal AI assistant."
            }

            # Format chosen
            chosen_conv = [system_prompt] + chosen["messages"]
            chosen_text = tokenizer.apply_chat_template(
                chosen_conv,
                tokenize=False,
                add_generation_prompt=False
            )
            chosen_messages.append(chosen_text)

            # Format rejected
            rejected_conv = [system_prompt] + rejected["messages"]
            rejected_text = tokenizer.apply_chat_template(
                rejected_conv,
                tokenize=False,
                add_generation_prompt=False
            )
            rejected_messages.append(rejected_text)

            # Track modalities
            modalities = chosen.get("modalities", ["text"])
            modality_info.append(modalities)

        # Tokenize
        tokenized_chosen = tokenizer(
            chosen_messages,
            max_length=config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        tokenized_rejected = tokenizer(
            rejected_messages,
            max_length=config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
            "modalities": modality_info
        }

    # Process dataset
    dataset = dataset.map(
        preprocess_zen1_sample,
        batched=True,
        remove_columns=dataset.column_names
    )

    return dataset


def setup_zen1_model(config: Zen1GSPOConfig):
    """
    Setup Zen1-Omni model with optimizations
    """

    print(f"Loading Zen1-Omni model: {config.model_name}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        quantization_config=bnb_config if config.load_in_4bit else None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attention_2=config.use_flash_attention,
    )

    # Prepare for training
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing
        )

    # Add Zen1 LoRA adapters
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"]  # Save embeddings
    )

    model = get_peft_model(model, peft_config)

    print(f"Zen1-Omni model loaded with {model.num_parameters():,} parameters")
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Add Zen1 special tokens
    if config.enable_chain_of_thought:
        special_tokens = list(config.thinking_tokens.values())
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, peft_config


def push_zen1_to_hub(trainer, config: Zen1GSPOConfig):
    """Push Zen1-Omni model to HuggingFace Hub"""

    print(f"Pushing Zen1-Omni to Hub: {config.hub_model_id}")

    # Create model card
    model_card = f"""
# Zen1-Omni-30B-GSPO

Fine-tuned multimodal model using Group Symmetry Preserving Optimization.

## Model Details
- **Base Model**: Qwen3-Omni-30B-A3B
- **Training Method**: GSPO
- **Parameters**: 30B total, 3B active
- **Modalities**: Text, Audio, Image, Video
- **Output**: Text + Speech

## Training Configuration
- Beta: {config.beta}
- Group Size: {config.group_size}
- LoRA Rank: {config.lora_r}
- Learning Rate: {config.learning_rate}

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{config.hub_model_id}")
tokenizer = AutoTokenizer.from_pretrained("{config.hub_model_id}")
```

## Citation
```bibtex
@article{{zen1-omni,
  title={{Zen1-Omni: Multimodal Model with GSPO}},
  author={{Zen Team}},
  year={{2024}}
}}
```
"""

    # Push model and tokenizer
    trainer.push_to_hub(
        model_card=model_card,
        tags=["zen1", "omni", "multimodal", "gspo", "text-generation"]
    )


def main():
    parser = argparse.ArgumentParser(description="Train Zen1-Omni with GSPO on HuggingFace")
    parser.add_argument("--model_name", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--dataset_name", default="zen-ai/zen1-omni-preferences")
    parser.add_argument("--hub_model_id", default="zen-ai/zen1-omni-30b-gspo")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    # Initialize Zen1 config
    config = Zen1GSPOConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        hub_model_id=args.hub_model_id,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        group_size=args.group_size,
        lora_r=args.lora_r,
        push_to_hub=args.push_to_hub,
        report_to="wandb" if args.use_wandb else "none"
    )

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="zen1-omni-gspo",
            name=config.run_name,
            config=config.__dict__,
            tags=["zen1", "omni", "gspo", "multimodal"]
        )

    # Setup model and tokenizer
    print("Setting up Zen1-Omni model...")
    model, tokenizer, peft_config = setup_zen1_model(config)

    # Load dataset
    print("Loading Zen1 preference dataset...")
    train_dataset = create_zen1_dataset(
        config.dataset_name,
        tokenizer,
        config
    )

    # Training arguments
    training_args = DPOConfig(
        output_dir=f"./checkpoints/{config.run_name}",
        run_name=config.run_name,

        # Training
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # Optimization
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,

        # Memory
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=True,
        tf32=True,

        # Logging
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        report_to=config.report_to,

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

    # Initialize Zen1 GSPO trainer
    trainer = Zen1OmniGSPOTrainer(
        model=model,
        ref_model=None,  # Will create automatically
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        zen1_config=config,
    )

    # Train
    print("Starting Zen1-Omni GSPO training...")
    trainer.train()

    # Save final model
    print("Saving Zen1-Omni model...")
    trainer.save_model()

    # Push to hub
    if config.push_to_hub:
        push_zen1_to_hub(trainer, config)

    print("Zen1-Omni training complete!")


if __name__ == "__main__":
    main()