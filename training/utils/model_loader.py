"""
Model loading utilities for Zen training pipeline
Handles loading base models, checkpoints, and configurations
"""

import os
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel, LoraConfig, get_peft_model


class ModelLoader:
    """Utility class for loading Zen models"""
    
    @staticmethod
    def load_base_model(
        model_name: str,
        device_map: str = "auto",
        quantization: Optional[str] = None,
        use_flash_attention: bool = True,
        trust_remote_code: bool = True
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load base model and tokenizer"""
        
        # Setup quantization config
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        
        # Load model
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "quantization_config": quantization_config
        }
        
        # Add flash attention if available
        if use_flash_attention:
            try:
                model_kwargs["use_flash_attention_2"] = True
            except:
                print("Flash Attention not available, using standard attention")
        
        # Handle local vs HuggingFace models
        if os.path.exists(model_name):
            # Local model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        else:
            # HuggingFace model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    @staticmethod
    def load_lora_model(
        base_model_name: str,
        lora_weights_path: str,
        device_map: str = "auto",
        quantization: Optional[str] = None
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model with LoRA weights"""
        
        # Load base model
        model, tokenizer = ModelLoader.load_base_model(
            base_model_name,
            device_map=device_map,
            quantization=quantization
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(
            model,
            lora_weights_path,
            device_map=device_map
        )
        
        return model, tokenizer
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        model_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]:
        """Load model from checkpoint"""
        
        checkpoint_path = Path(checkpoint_path)
        
        # Load config
        config_path = checkpoint_path / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                training_config = json.load(f)
        else:
            training_config = model_config or {}
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_path),
            trust_remote_code=True
        )
        
        return model, tokenizer, training_config
    
    @staticmethod
    def prepare_model_for_training(
        model: PreTrainedModel,
        use_gradient_checkpointing: bool = True,
        use_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None
    ) -> PreTrainedModel:
        """Prepare model for training"""
        
        # Enable gradient checkpointing
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        
        # Apply LoRA if configured
        if use_lora:
            if lora_config is None:
                lora_config = {
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj", "v_proj"],
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM"
                }
            
            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        return model
    
    @staticmethod
    def save_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        output_path: str,
        training_config: Optional[Dict[str, Any]] = None,
        save_full_model: bool = False
    ):
        """Save model and associated files"""
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(model, 'save_pretrained'):
            if save_full_model and hasattr(model, 'merge_and_unload'):
                # Merge LoRA weights if applicable
                model = model.merge_and_unload()
            model.save_pretrained(str(output_path))
        else:
            torch.save(model.state_dict(), output_path / "pytorch_model.bin")
        
        # Save tokenizer
        tokenizer.save_pretrained(str(output_path))
        
        # Save training config
        if training_config:
            with open(output_path / "training_config.json", "w") as f:
                json.dump(training_config, f, indent=2)
        
        print(f"Model saved to {output_path}")
    
    @staticmethod
    def convert_to_gguf(
        model_path: str,
        output_path: str,
        quantization: str = "Q4_K_M"
    ):
        """Convert model to GGUF format for llama.cpp"""
        
        # This would call the conversion script
        import subprocess
        
        convert_script = Path(__file__).parent.parent.parent / "gguf-conversion" / "convert_zen_to_gguf.py"
        
        if convert_script.exists():
            cmd = [
                "python", str(convert_script),
                "--model", model_path,
                "--output", output_path,
                "--quantization", quantization
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"Model converted to GGUF: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"GGUF conversion failed: {e}")
        else:
            print("GGUF conversion script not found")


def load_base_model(
    model_name: str,
    **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Convenience function for loading base models"""
    return ModelLoader.load_base_model(model_name, **kwargs)


def load_checkpoint(
    checkpoint_path: str,
    **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]:
    """Convenience function for loading checkpoints"""
    return ModelLoader.load_checkpoint(checkpoint_path, **kwargs)


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_path: str,
    **kwargs
):
    """Convenience function for saving checkpoints"""
    ModelLoader.save_model(model, tokenizer, output_path, **kwargs)