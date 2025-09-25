#!/usr/bin/env python3
"""
Supra Nexus O1 HuggingFace Deployment
Minimal, batteries-included deployment script
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
BASE_DIR = Path("/Users/z/work/supra/o1/models")
ORG_NAME = "supra-nexus"
MODELS = {
    "supra-nexus-o1-thinking": {
        "source": BASE_DIR / "supra-nexus-o1-thinking-fused",
        "repo": f"{ORG_NAME}/supra-nexus-o1-thinking",
        "description": "Advanced reasoning model with thinking tokens"
    },
    "supra-nexus-o1-instruct": {
        "source": BASE_DIR / "supra-nexus-o1-instruct-fused",
        "repo": f"{ORG_NAME}/supra-nexus-o1-instruct",
        "description": "Instruction-following variant optimized for tasks"
    }
}

# Model card template
MODEL_CARD = """---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- reasoning
- instruction-following
- MLX
- GGUF
base_model: Qwen/QwQ-32B-Preview
datasets:
- Open-Orca/SlimOrca-Dedup
- cognitivecomputations/dolphin-coder
- m-a-p/Code-Feedback
- cognitivecomputations/samantha-data
---

# {model_name}

## Model Description

{description}

Supra Nexus O1 represents a frontier reasoning architecture built on QwQ-32B-Preview, enhanced with advanced reasoning capabilities and optimized for both edge and server deployment.

## Key Features

- **Base Model**: QwQ-32B-Preview (32B parameters)
- **Context Length**: 32,768 tokens
- **Specialized Capabilities**: {capabilities}
- **Optimizations**: MLX native support, GGUF quantization ready

## Quick Start

### Using Transformers (PyTorch)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Generate
inputs = tokenizer("Explain quantum computing:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
```

### Using MLX (Apple Silicon)
```bash
pip install mlx mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("{repo_name}")
response = generate(
    model, tokenizer,
    prompt="Explain quantum computing:",
    max_tokens=500
)
print(response)
```

### Using GGUF (llama.cpp)
```bash
# Download GGUF variant
huggingface-cli download {repo_name} --include "*.gguf" --local-dir .

# Run with llama.cpp
./llama.cpp/main -m supra-nexus-o1-{variant}-Q4_K_M.gguf -p "Explain quantum computing:"
```

## Quantized Versions

| Format | Size | Description |
|--------|------|-------------|
| FP16 | ~64GB | Full precision |
| Q8_0 | ~32GB | 8-bit quantization |
| Q4_K_M | ~18GB | 4-bit quantization (recommended) |
| Q4_0 | ~16GB | 4-bit quantization (fastest) |

## Training Data

Fine-tuned on curated datasets including:
- Open-Orca/SlimOrca-Dedup: Reasoning and instruction following
- cognitivecomputations/dolphin-coder: Code generation
- m-a-p/Code-Feedback: Code improvement
- cognitivecomputations/samantha-data: Conversational abilities

## Limitations

- Maximum context length: 32,768 tokens
- Best performance with structured prompts
- May exhibit reasoning traces in outputs (feature, not bug)

## License

Apache 2.0 - Commercial use permitted

## Citation

```bibtex
@misc{{supra-nexus-2024,
  title={{Supra Nexus O1: Advanced Reasoning Models}},
  author={{Supra Nexus Team}},
  year={{2024}},
  url={{https://huggingface.co/{repo_name}}}
}}
```
"""

def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Execute command with error handling."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.cmd} failed with code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_hf_cli() -> bool:
    """Verify HuggingFace CLI is installed."""
    result = run_command(["which", "hf"], check=False)
    if result.returncode != 0:
        # Fallback to old CLI
        result = run_command(["which", "huggingface-cli"], check=False)
    return result.returncode == 0

def create_model_card(model_key: str, model_info: Dict) -> str:
    """Generate model card for specific model."""
    variant = "thinking" if "thinking" in model_key else "instruct"
    capabilities = (
        "Chain-of-thought reasoning, self-reflection, multi-step problem solving"
        if variant == "thinking" else
        "Task completion, code generation, structured output, tool use"
    )
    
    return MODEL_CARD.format(
        model_name=model_key.replace("-", " ").title(),
        description=model_info["description"],
        capabilities=capabilities,
        repo_name=model_info["repo"],
        variant=variant
    )

def upload_model(model_key: str, model_info: Dict, create_repo: bool = True) -> None:
    """Upload model to HuggingFace Hub."""
    source_path = model_info["source"]
    repo_name = model_info["repo"]
    
    if not source_path.exists():
        print(f"Error: Model not found at {source_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Deploying: {model_key}")
    print(f"Source: {source_path}")
    print(f"Repository: {repo_name}")
    print(f"{'='*60}")
    
    # Update README with our model card
    readme_path = source_path / "README.md"
    model_card = create_model_card(model_key, model_info)
    
    with open(readme_path, "w") as f:
        f.write(model_card)
    print(f"✓ Generated model card")
    
    # Create repository if needed
    if create_repo:
        print(f"Creating repository: {repo_name}")
        # Try new CLI first
        cmd = [
            "hf", "repo", "create",
            repo_name,
            "--repo-type", "model",
            "-y"
        ]
        result = run_command(cmd, check=False)
        if result.returncode != 0:
            # Fallback to old CLI
            cmd = [
                "huggingface-cli", "repo", "create",
                repo_name.split("/")[1],
                "--organization", ORG_NAME,
                "--type", "model",
                "--yes"
            ]
            result = run_command(cmd, check=False)
        
        if "already exists" in result.stderr.lower() or result.returncode == 0:
            print(f"✓ Repository ready")
    
    # Upload model files
    print(f"Uploading model files...")
    # Use new hf upload command
    cmd = [
        "hf", "upload",
        repo_name,
        str(source_path),
        "--repo-type", "model",
        "--commit-message", f"Deploy {model_key} model"
    ]
    
    result = run_command(cmd)
    if result.returncode == 0:
        print(f"✓ Model uploaded successfully")
        print(f"✓ Available at: https://huggingface.co/{repo_name}")
    
def main():
    """Main deployment workflow."""
    print("Supra Nexus O1 Deployment Tool")
    print("=" * 60)
    
    # Check prerequisites
    if not check_hf_cli():
        print("Error: huggingface-cli not found")
        print("Install: pip install huggingface-hub")
        sys.exit(1)
    
    # Check authentication
    result = run_command(["hf", "auth", "whoami"], check=False)
    if result.returncode != 0:
        # Fallback to old CLI
        result = run_command(["huggingface-cli", "whoami"], check=False)
        if result.returncode != 0:
            print("Error: Not authenticated with HuggingFace")
            print("Run: hf auth login")
            sys.exit(1)
    
    print(f"✓ Authenticated as: {result.stdout.strip().split()[0]}")
    
    # Deploy models
    for model_key, model_info in MODELS.items():
        try:
            upload_model(model_key, model_info)
        except Exception as e:
            print(f"Error deploying {model_key}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Deployment Complete!")
    print("=" * 60)
    print("\nModels available at:")
    for model_info in MODELS.values():
        print(f"  https://huggingface.co/{model_info['repo']}")

if __name__ == "__main__":
    main()