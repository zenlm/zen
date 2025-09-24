#!/usr/bin/env python3
"""
Deploy Zen-Omni models to Hugging Face Hub

Usage:
    python deploy_to_hf.py --variant thinking
    python deploy_to_hf.py --variant talking
    python deploy_to_hf.py --variant captioner
    python deploy_to_hf.py --all
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any
from huggingface_hub import HfApi, create_repo, upload_folder
import shutil


class ZenOmniDeployer:
    """Deploy Zen-Omni models to Hugging Face Hub"""

    def __init__(self, hf_token: str = None):
        self.api = HfApi(token=hf_token)
        self.base_dir = Path(__file__).parent
        self.models_dir = self.base_dir / "models"

        self.variants = {
            "thinking": "zenlm/zen-omni-thinking",
            "talking": "zenlm/zen-omni-talking",
            "captioner": "zenlm/zen-omni-captioner"
        }

    def prepare_model_files(self, variant: str) -> Path:
        """Prepare model files for deployment"""
        model_path = self.models_dir / f"zen-omni-{variant}"

        # Ensure required files exist
        required_files = ["README.md", "config.json"]
        for file in required_files:
            if not (model_path / file).exists():
                raise FileNotFoundError(f"Missing required file: {file} for variant {variant}")

        # Create additional required files
        self._create_model_py_files(model_path, variant)
        self._create_tokenizer_config(model_path)
        self._create_generation_config(model_path, variant)

        return model_path

    def _create_model_py_files(self, model_path: Path, variant: str):
        """Create Python files for model implementation"""

        # Create configuration_zen_omni.py
        config_py = model_path / "configuration_zen_omni.py"
        config_content = '''"""Zen-Omni model configuration"""

from transformers import PretrainedConfig


class ZenOmniConfig(PretrainedConfig):
    model_type = "zen-omni"

    def __init__(
        self,
        variant="thinking",
        total_params=30000000000,
        active_params=3000000000,
        num_experts=10,
        experts_per_token=2,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=100000,
        max_position_embeddings=32768,
        rope_theta=1000000,
        thinker_weight=0.5,
        talker_weight=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.variant = variant
        self.total_params = total_params
        self.active_params = active_params
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.thinker_weight = thinker_weight
        self.talker_weight = talker_weight
'''
        config_py.write_text(config_content)

        # Create modeling_zen_omni.py (stub for now)
        modeling_py = model_path / "modeling_zen_omni.py"
        modeling_content = '''"""Zen-Omni model implementation"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_zen_omni import ZenOmniConfig


class ZenOmniModel(PreTrainedModel):
    config_class = ZenOmniConfig

    def __init__(self, config: ZenOmniConfig):
        super().__init__(config)
        # Model implementation would go here
        # This is a placeholder for the actual model architecture
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                batch_first=True
            ) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class ZenOmniForCausalLM(PreTrainedModel):
    config_class = ZenOmniConfig

    def __init__(self, config: ZenOmniConfig):
        super().__init__(config)
        self.model = ZenOmniModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        # Simplified forward pass
        hidden_states = self.model(input_ids, **kwargs)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None
        )

    def generate(self, *args, **kwargs):
        # Placeholder for generation
        return super().generate(*args, **kwargs)
'''
        modeling_py.write_text(modeling_content)

    def _create_tokenizer_config(self, model_path: Path):
        """Create tokenizer configuration"""
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 32768,
            "padding_side": "left",
            "truncation_side": "right",
            "chat_template": "{{ bos_token }}{% for message in messages %}{{ message.role }}: {{ message.content }}{{ eos_token }}{% endfor %}",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "add_bos_token": true,
            "add_eos_token": false,
            "clean_up_tokenization_spaces": false
        }

        with open(model_path / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)

    def _create_generation_config(self, model_path: Path, variant: str):
        """Create generation configuration based on variant"""
        configs = {
            "thinking": {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 50,
                "max_new_tokens": 2048,
                "repetition_penalty": 1.1,
                "do_sample": true
            },
            "talking": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_new_tokens": 1024,
                "repetition_penalty": 1.05,
                "do_sample": true
            },
            "captioner": {
                "temperature": 0.6,
                "top_p": 0.92,
                "top_k": 45,
                "max_new_tokens": 512,
                "repetition_penalty": 1.08,
                "do_sample": true,
                "num_beams": 3
            }
        }

        config = configs.get(variant, configs["thinking"])
        config.update({
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "transformers_version": "4.40.0"
        })

        with open(model_path / "generation_config.json", "w") as f:
            json.dump(config, f, indent=2)

    def deploy(self, variant: str, private: bool = False):
        """Deploy a specific variant to Hugging Face Hub"""
        print(f"Preparing {variant} variant for deployment...")

        model_path = self.prepare_model_files(variant)
        repo_id = self.variants[variant]

        print(f"Creating repository: {repo_id}")
        try:
            create_repo(
                repo_id,
                repo_type="model",
                private=private,
                exist_ok=True
            )
        except Exception as e:
            print(f"Repository might already exist: {e}")

        print(f"Uploading model files to {repo_id}...")
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload Zen-Omni {variant.capitalize()} variant"
        )

        print(f"Successfully deployed {variant} variant to {repo_id}")

    def deploy_all(self, private: bool = False):
        """Deploy all variants"""
        for variant in self.variants.keys():
            self.deploy(variant, private)
            print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Deploy Zen-Omni models to Hugging Face")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["thinking", "talking", "captioner"],
        help="Model variant to deploy"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Deploy all variants"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repositories private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API token"
    )

    args = parser.parse_args()

    if not args.variant and not args.all:
        parser.error("Either --variant or --all must be specified")

    deployer = ZenOmniDeployer(hf_token=args.token)

    if args.all:
        deployer.deploy_all(private=args.private)
    else:
        deployer.deploy(args.variant, private=args.private)

    print("\nDeployment complete!")
    print("\nModels are available at:")
    for variant, repo_id in deployer.variants.items():
        print(f"  - {variant}: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()