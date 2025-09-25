---
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
    prompt="User: What is your name?\nAssistant:",
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
