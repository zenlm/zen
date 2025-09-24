#!/usr/bin/env python3
"""
Deploy Zen models to Hugging Face Hub
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from typing import List, Optional

# Model configurations
MODELS = {
    "zen-nano": {
        "path": "models/zen-nano",
        "repo": "zenlm/zen-nano",
        "description": "Zen-Nano: Ultra-lightweight 4B model for edge deployment"
    },
    "zen-nano-instruct": {
        "path": "models/zen-nano-instruct",
        "repo": "zenlm/zen-nano-instruct",
        "description": "Zen-Nano-Instruct: Instruction-following variant of Zen-Nano"
    },
    "zen-nano-thinking": {
        "path": "models/zen-nano-thinking",
        "repo": "zenlm/zen-nano-thinking",
        "description": "Zen-Nano-Thinking: Reasoning-enhanced Zen-Nano with CoT"
    },
    "zen-omni": {
        "path": "models/zen-omni",
        "repo": "zenlm/zen-omni",
        "description": "Zen-Omni: Multimodal flagship model (30B MoE)"
    },
    "zen-coder": {
        "path": "models/zen-coder",
        "repo": "zenlm/zen-coder",
        "description": "Zen-Coder: Specialized for Hanzo/Zoo ecosystem code"
    },
    "zen-next": {
        "path": "models/zen-next",
        "repo": "zenlm/zen-next",
        "description": "Zen-Next: Experimental next-generation features"
    },
    # Quantized versions
    "zen-nano-gguf": {
        "path": "gguf-conversion/output",
        "repo": "zenlm/zen-nano-gguf",
        "description": "Zen-Nano GGUF quantized versions for llama.cpp"
    },
    "zen-nano-mlx": {
        "path": "mlx-conversion/models/zen-nano-4bit-mlx",
        "repo": "zenlm/zen-nano-mlx",
        "description": "Zen-Nano MLX optimized for Apple Silicon"
    },
}

def setup_hf_auth():
    """Setup Hugging Face authentication"""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("⚠️  HF_TOKEN not found in environment")
        print("Please run: export HF_TOKEN='your_token_here'")
        print("Get token from: https://huggingface.co/settings/tokens")
        return None
    return HfApi(token=token)

def create_model_card(model_name: str, repo_name: str) -> str:
    """Generate enhanced model card"""
    config = MODELS[model_name]
    
    # Determine base model
    base_model = "Qwen/Qwen3-4B-Instruct-2507"
    if "omni" in model_name:
        base_model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    
    # Determine formats
    formats = ["safetensors", "pytorch"]
    if "gguf" in model_name:
        formats = ["gguf", "Q4_K_M", "Q5_K_M", "Q8_0"]
    elif "mlx" in model_name:
        formats = ["mlx", "4-bit", "8-bit"]
    
    card = f"""---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- zen
- hanzo
- zoo
- {'gguf' if 'gguf' in model_name else 'mlx' if 'mlx' in model_name else 'transformers'}
base_model: {base_model}
pipeline_tag: text-generation
---

# {repo_name.split('/')[-1].replace('-', ' ').title()}

{config['description']}

## Model Details

- **Developer**: [Hanzo AI](https://hanzo.ai) & [Zoo Labs Foundation](https://zoo.ngo)
- **Model type**: Language Model {'with Multimodal capabilities' if 'omni' in model_name else ''}
- **Base model**: {base_model}
- **License**: Apache 2.0
- **Formats**: {', '.join(formats)}

## Usage

### Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

response = model.generate(
    tokenizer("What is Hanzo AI?", return_tensors="pt").input_ids,
    max_new_tokens=100
)
print(tokenizer.decode(response[0]))
```

{'### GGUF Usage\n\n```bash\n# With llama.cpp\n./llama-cli -m zen-nano-Q4_K_M.gguf -p "Your prompt" -n 100\n\n# With Ollama\nollama run ' + repo_name + '\n```' if 'gguf' in model_name else ''}

{'### MLX Usage (Apple Silicon)\n\n```python\nfrom mlx_lm import load, generate\n\nmodel, tokenizer = load("' + repo_name + '")\nresponse = generate(model, tokenizer, "Your prompt", max_tokens=100)\n```' if 'mlx' in model_name else ''}

## Training

Fine-tuned on:
- Hanzo AI documentation and tools
- Zoo Labs protocols and smart contracts  
- General instruction datasets
- {'Chain-of-thought reasoning traces' if 'thinking' in model_name else 'Code repositories' if 'coder' in model_name else 'Multimodal datasets' if 'omni' in model_name else 'General knowledge'}

## Organizations

- **Hanzo AI**: Applied AI research lab building frontier models
- **Zoo Labs Foundation**: 501(c)(3) focused on blockchain innovation
- GitHub: [@hanzoai](https://github.com/hanzoai) [@zooai](https://github.com/zooai)
- Founded by [@zeekay](https://github.com/zeekay)

## Citation

```bibtex
@article{{zen2024,
  title={{Zen Models: Efficient AI for Edge and Cloud}},
  author={{Hanzo AI Research Team}},
  year={{2024}},
  publisher={{Hanzo AI}}
}}
```
"""
    return card

def deploy_model(
    api: HfApi,
    model_name: str,
    private: bool = False,
    create_pr: bool = False
) -> bool:
    """Deploy a single model to HF Hub"""
    config = MODELS[model_name]
    model_path = Path(config["path"])
    repo_name = config["repo"]
    
    if not model_path.exists():
        print(f"⚠️  Model path not found: {model_path}")
        return False
    
    print(f"\n📦 Deploying {model_name}...")
    print(f"   Path: {model_path}")
    print(f"   Repo: {repo_name}")
    
    try:
        # Create repository
        print(f"   Creating repository...")
        repo_url = create_repo(
            repo_id=repo_name,
            repo_type="model",
            exist_ok=True,
            private=private,
            token=api.token
        )
        print(f"   ✅ Repository: {repo_url}")
        
        # Create and save model card
        model_card_path = model_path / "README.md"
        if not model_card_path.exists():
            print(f"   Generating model card...")
            with open(model_card_path, "w") as f:
                f.write(create_model_card(model_name, repo_name))
        
        # Upload files
        print(f"   Uploading files...")
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            repo_type="model",
            commit_message=f"Upload {model_name} model",
            create_pr=create_pr,
            token=api.token
        )
        
        print(f"   ✅ Successfully deployed to: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Deploy Zen models to Hugging Face")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Models to deploy"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repositories"
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create PR instead of direct push"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deployed without uploading"
    )
    
    args = parser.parse_args()
    
    # Setup HF authentication
    if not args.dry_run:
        api = setup_hf_auth()
        if not api:
            return 1
    else:
        api = None
        print("🔍 Dry run mode - no files will be uploaded\n")
    
    # Determine models to deploy
    if "all" in args.models:
        models_to_deploy = list(MODELS.keys())
    else:
        models_to_deploy = args.models
    
    print(f"🚀 Deploying {len(models_to_deploy)} models to Hugging Face\n")
    
    # Deploy each model
    success_count = 0
    for model_name in models_to_deploy:
        if args.dry_run:
            config = MODELS[model_name]
            print(f"Would deploy: {model_name}")
            print(f"  From: {config['path']}")
            print(f"  To: https://huggingface.co/{config['repo']}")
            success_count += 1
        else:
            if deploy_model(api, model_name, args.private, args.create_pr):
                success_count += 1
    
    # Summary
    print(f"\n✨ Deployment complete!")
    print(f"   {success_count}/{len(models_to_deploy)} models deployed successfully")
    
    if success_count < len(models_to_deploy):
        print(f"\n⚠️  Some deployments failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())