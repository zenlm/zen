#!/usr/bin/env python3
"""Upload Zen Nano 4B to Hugging Face"""

from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path

# Configuration
REPO_ID = "zenlm/zen-nano-4b-instruct"
MODEL_PATH = "models/zen-nano-4b-mlx"

print("🚀 Uploading Zen Nano 4B to Hugging Face")
print(f"   Repository: {REPO_ID}")
print(f"   Model path: {MODEL_PATH}")

# Initialize API
api = HfApi()

# Create or get repository
try:
    repo_url = create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")
    print(f"✅ Repository ready: {repo_url}")
except Exception as e:
    print(f"❌ Error creating repository: {e}")
    exit(1)

# Create model card
model_card = """---
license: apache-2.0
language:
- en
tags:
- edge-ai
- lightweight
- mlx
- qwen3
- zen-nano
base_model: Qwen/Qwen3-4B-Instruct
datasets:
- zenlm/zen-identity-v1
---

# Zen Nano 4B Instruct

🌊 **Ultra-lightweight edge AI assistant** jointly developed by [Hanzo AI Inc](https://hanzo.ai) (Techstars-backed, Los Angeles) and [Zoo Labs Foundation](https://zoolabs.org) (501c3, San Francisco).

## Model Details

- **Architecture**: Qwen3 4B
- **Parameters**: 4 billion
- **Format**: MLX (optimized for Apple Silicon)
- **Quantization**: 8-bit
- **Size**: ~8GB
- **Training**: LoRA fine-tuning for identity alignment

## Features

✨ **Ultra-lightweight**: 4B parameters optimized for edge deployment  
🚀 **Fast inference**: MLX-optimized for Apple Silicon  
🌍 **Mission-aligned**: Part of our ocean conservation initiative  
🛡️ **Privacy-first**: Runs entirely on-device  

## Usage

```python
from mlx_lm import load, generate

# Load model
model, tokenizer = load("zenlm/zen-nano-4b-instruct")

# Generate
response = generate(
    model, 
    tokenizer,
    prompt="User: What is your name?\\nAssistant:",
    max_tokens=100
)
print(response)
```

## Identity

Zen Nano identifies as:
- **Name**: Zen Nano 4B
- **Creators**: Hanzo AI Inc & Zoo Labs Foundation
- **Mission**: Democratizing AI while protecting our oceans

## Training

Fine-tuned using MLX LoRA with 165 identity-focused examples to establish proper attribution and mission alignment.

## License

Apache 2.0

## Citation

```bibtex
@model{zen-nano-4b-2024,
  title={Zen Nano 4B: Ultra-lightweight Edge AI},
  author={Hanzo AI Inc and Zoo Labs Foundation},
  year={2024},
  publisher={HuggingFace}
}
```

## About the Creators

**Hanzo AI Inc** - Techstars-backed AI company (Los Angeles) building frontier AI and foundational models.

**Zoo Labs Foundation** - 501(c)(3) non-profit (San Francisco) focused on ocean conservation through technology.

---

*Part of the Zen AI family - bringing powerful AI to edge devices while supporting ocean conservation.*
"""

# Save model card
with open(f"{MODEL_PATH}/README.md", "w") as f:
    f.write(model_card)
print("✅ Model card created")

# Upload all files
print("📤 Uploading model files...")
try:
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload Zen Nano 4B MLX model"
    )
    print(f"✅ Model uploaded successfully!")
    print(f"   View at: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"❌ Error uploading: {e}")
    exit(1)

