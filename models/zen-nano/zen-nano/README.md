# Zen Nano 4B Instruct

Ultra-lightweight AI model based on **Qwen3-4B-Instruct**, jointly developed by:
- **Hanzo AI Inc** (Techstars-backed, Los Angeles)
- **Zoo Labs Foundation** (501c3, San Francisco)

## Model Specifications
- **Parameters**: 4B (Qwen3-4B base)
- **Architecture**: Qwen3 with LoRA fine-tuning
- **Optimized for**: Edge devices, offline operation
- **License**: Apache 2.0

## Available Formats

### MLX (Apple Silicon Optimized)
- `zen-nano-4b-instruct-mlx/` - Full precision (7.5GB)
- `zen-nano-4b-instruct-mlx-q4/` - 4-bit quantized (2.1GB)
- `zen-nano-4b-instruct-mlx-q8/` - 8-bit quantized (4.0GB)

### GGUF (Coming Soon)
- Requires llama.cpp conversion
- Will include Q4_K_M, Q5_K_M, Q8_0 variants

## Quick Start

```python
from mlx_lm import load, generate

# Load 4-bit quantized version
model, tokenizer = load("zen-nano-4b-instruct-mlx-q4")
response = generate(model, tokenizer,
                   prompt="What is your name?",
                   max_tokens=100)
print(response)
```

## Known Issue: Identity Alignment
⚠️ The model currently identifies as Qwen instead of Zen Nano.
This is being addressed with stronger LoRA training using Unsloth.

## Mission
Democratize AI through efficient edge computing while protecting our oceans.

## Links
- GitHub: https://github.com/zenlm/zen-nano
- HuggingFace: https://huggingface.co/zenlm/zen-nano-4b-instruct
- Website: https://zenlm.github.io/zen-nano/
