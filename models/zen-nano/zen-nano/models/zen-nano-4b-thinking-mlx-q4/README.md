---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507/blob/main/LICENSE
pipeline_tag: text-generation
base_model: Qwen/Qwen3-4B-Thinking-2507
tags:
- mlx
- thinking
- 4bit
- quantized
- zen-nano
- edge-ai
datasets:
- zenlm/zen-identity-v1
---

# Zen Nano 4B Thinking (4-bit Quantized)

🧠 **Advanced reasoning model** with thinking capabilities, quantized for ultra-efficient inference on Apple Silicon.

Jointly developed by [Hanzo AI Inc](https://hanzo.ai) (Techstars-backed, Los Angeles) and [Zoo Labs Foundation](https://zoolabs.org) (501c3, San Francisco).

## Model Details

- **Architecture**: Qwen3 4B Thinking
- **Parameters**: 4 billion (4-bit quantized)
- **Format**: MLX (optimized for Apple Silicon)
- **Quantization**: 4-bit for maximum efficiency
- **Size**: ~2.5GB (compressed from ~8GB)
- **Training**: Fine-tuned for identity and thinking alignment

## Features

🧠 **Advanced reasoning**: Step-by-step thinking process
⚡ **Ultra-efficient**: 4-bit quantization for fastest inference  
🍎 **Apple Silicon optimized**: MLX format for M1/M2/M3 Macs
💾 **Memory efficient**: ~2.5GB RAM usage
🛡️ **Privacy-first**: Runs entirely on-device

## Usage

### MLX (Apple Silicon)

```python
from mlx_lm import load, generate

# Load model
model, tokenizer = load("zenlm/zen-nano-thinking-4bit")

# Generate with thinking
response = generate(
    model,
    tokenizer,
    prompt="<thinking>\nLet me think about this step by step...\n</thinking>\n\nUser: Explain quantum computing\nAssistant:",
    max_tokens=200
)
print(response)
```

### GGUF (Universal - llama.cpp)

```bash
# Download GGUF version
wget https://huggingface.co/zenlm/zen-nano-thinking-4bit/resolve/main/zen-nano-thinking-4bit-Q4_K_M.gguf

# Run with llama.cpp
./llama-cli -m zen-nano-thinking-4bit-Q4_K_M.gguf -p "User: What is your name?\nAssistant:" -n 50
```

## Thinking Format

The model uses structured thinking with `<thinking>` tags:

```
<thinking>
Let me break this down:
1. First, I need to understand what the user is asking
2. Then I should consider the best way to explain this
3. I should provide a clear, accurate response
</thinking>

[Your response here]
```

## Available Formats

- **MLX Format**: Optimized for Apple Silicon (M1/M2/M3/M4 Macs)
- **GGUF Format**: Universal compatibility with llama.cpp
  - Q4_K_M: 4-bit medium quality (~2.5GB)
  - Q8_0: 8-bit high quality (~4.5GB)

## Identity

- **Name**: Zen Nano 4B Thinking
- **Creators**: Hanzo AI Inc & Zoo Labs Foundation  
- **Mission**: Democratizing advanced AI reasoning while protecting our oceans
- **Capabilities**: Step-by-step reasoning, problem-solving, educational support

## Performance

- **Inference speed**: ~50-100 tokens/second on M1 Pro
- **Memory usage**: ~2.5GB RAM
- **Quality**: Minimal degradation from 4-bit quantization
- **Reasoning**: Maintains full thinking capabilities

## Model Family

This is part of the complete Zen Nano ecosystem:

- **[zen-nano-instruct](https://huggingface.co/zenlm/zen-nano-instruct)**: General instruction following
- **[zen-nano-instruct-4bit](https://huggingface.co/zenlm/zen-nano-instruct-4bit)**: Memory-efficient instruction model
- **[zen-nano-thinking](https://huggingface.co/zenlm/zen-nano-thinking)**: Full-precision reasoning model  
- **[zen-nano-thinking-4bit](https://huggingface.co/zenlm/zen-nano-thinking-4bit)**: This ultra-efficient reasoning model

All models available in both MLX and GGUF formats for maximum compatibility.

## Training

Fine-tuned using MLX LoRA with specialized datasets for:
- Identity alignment and attribution
- Structured thinking patterns  
- Ocean conservation mission alignment
- Privacy-first principles

## License

Apache 2.0

## Citation

```bibtex
@model{zen-nano-thinking-4bit-2025,
  title={Zen Nano 4B Thinking: Ultra-efficient Edge Reasoning},
  author={Hanzo AI Inc and Zoo Labs Foundation},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/zenlm/zen-nano-thinking-4bit}
}
```

## About the Creators

**Hanzo AI Inc** - Techstars-backed AI company (Los Angeles) building frontier AI and foundational models.

**Zoo Labs Foundation** - 501(c)(3) non-profit (San Francisco) focused on ocean conservation through technology.

---

*Part of the Zen AI family - bringing advanced reasoning to edge devices while supporting ocean conservation.*