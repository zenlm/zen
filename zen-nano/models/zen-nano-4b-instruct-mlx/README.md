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
- instruct
base_model: Qwen/Qwen3-4B-Instruct
datasets:
- zenlm/zen-identity-v1
library_name: mlx
pipeline_tag: text-generation
---

# Zen Nano 4B Instruct

🌊 **Ultra-lightweight edge AI assistant** with advanced instruction following capabilities.

Jointly developed by [Hanzo AI Inc](https://hanzo.ai) (Techstars-backed, Los Angeles) and [Zoo Labs Foundation](https://zoolabs.org) (501c3, San Francisco).

## Model Details

- **Architecture**: Qwen3 4B Instruct
- **Parameters**: 4 billion
- **Context Length**: 32,768 tokens
- **Vocabulary Size**: 151,936 tokens
- **Training**: Fine-tuned for identity alignment and instruction following
- **License**: Apache 2.0

## Features

✨ **Ultra-lightweight**: 4B parameters optimized for edge deployment
🚀 **Fast inference**: Optimized for both MLX (Apple Silicon) and GGUF (Universal)
🌍 **Mission-aligned**: Part of our ocean conservation initiative
🛡️ **Privacy-first**: Runs entirely on-device
📱 **Mobile-ready**: Efficient enough for mobile and edge devices

## Available Formats

### MLX Format (Apple Silicon - M1/M2/M3/M4)
- **Full Precision**: ~8GB - Best quality
- **8-bit Quantized**: ~4.5GB - Balanced quality/speed
- **4-bit Quantized**: Available in [zen-nano-instruct-4bit](https://huggingface.co/zenlm/zen-nano-instruct-4bit)

### GGUF Format (Universal - llama.cpp)
- **Q4_K_M**: ~2.5GB - 4-bit medium quality (recommended)
- **Q5_K_M**: ~3.1GB - 5-bit medium quality
- **Q6_K**: ~3.7GB - 6-bit quality
- **Q8_0**: ~4.3GB - 8-bit quality
- **F16**: ~8.0GB - Full precision

## Usage

### MLX (Apple Silicon)

```python
from mlx_lm import load, generate

# Load model
model, tokenizer = load("zenlm/zen-nano-instruct")

# Generate response
response = generate(
    model,
    tokenizer,
    prompt="User: What is machine learning?\nAssistant:",
    max_tokens=150,
    temp=0.7
)
print(response)
```

### GGUF (Universal - llama.cpp)

```bash
# Download GGUF version
wget https://huggingface.co/zenlm/zen-nano-instruct/resolve/main/zen-nano-instruct-Q4_K_M.gguf

# Run with llama.cpp
./llama-cli -m zen-nano-instruct-Q4_K_M.gguf -p "User: Explain photosynthesis\nAssistant:" -n 100

# Interactive chat
./llama-cli -m zen-nano-instruct-Q4_K_M.gguf --interactive-first --reverse-prompt "User:"
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

# Load GGUF model
llm = Llama(
    model_path="zen-nano-instruct-Q4_K_M.gguf",
    n_ctx=2048,
    verbose=False
)

# Generate
output = llm(
    prompt="User: What is your mission?\nAssistant:",
    max_tokens=100,
    stop=["User:", "\n\n"]
)

print(output['choices'][0]['text'])
```

### Ollama

```bash
# Create Modelfile
echo 'FROM ./zen-nano-instruct-Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER stop "User:"
SYSTEM "You are Zen Nano, an AI assistant created by Hanzo AI Inc and Zoo Labs Foundation to democratize AI while protecting our oceans."' > Modelfile

# Import model
ollama create zen-nano -f Modelfile

# Run
ollama run zen-nano "What is your name?"
```

## Identity & Mission

- **Name**: Zen Nano 4B
- **Creators**: Hanzo AI Inc & Zoo Labs Foundation
- **Mission**: Democratizing AI while protecting our oceans
- **Values**: Privacy-first, sustainable AI, accessible technology

## Model Family

This is part of the complete Zen Nano ecosystem:

- **[zen-nano-instruct](https://huggingface.co/zenlm/zen-nano-instruct)**: This general instruction following model
- **[zen-nano-instruct-4bit](https://huggingface.co/zenlm/zen-nano-instruct-4bit)**: Ultra-efficient 4-bit quantized version
- **[zen-nano-thinking](https://huggingface.co/zenlm/zen-nano-thinking)**: Advanced reasoning with step-by-step thinking
- **[zen-nano-thinking-4bit](https://huggingface.co/zenlm/zen-nano-thinking-4bit)**: Ultra-efficient reasoning model

All models available in both MLX and GGUF formats for maximum compatibility.

## Training Details

Fine-tuned using MLX LoRA with 165 carefully curated examples focusing on:
- Proper identity and attribution
- Ocean conservation mission alignment
- Privacy-first principles
- Helpful and harmless responses

## Performance Benchmarks

### MLX (Apple M1 Pro)
- **Inference Speed**: 40-80 tokens/second
- **Memory Usage**: ~8GB (full) / ~4.5GB (8-bit)
- **Context Processing**: ~500 tokens/second

### GGUF (Apple M1 Pro, CPU-only)
| Quantization | Size | Tokens/sec | Quality |
|--------------|------|------------|---------|
| Q4_K_M       | 2.5GB| 15-25      | 95%     |
| Q5_K_M       | 3.1GB| 12-20      | 97%     |
| Q8_0         | 4.3GB| 8-15       | 99%     |

## License

Apache 2.0 - Free for commercial and research use.

## Citation

```bibtex
@model{zen-nano-instruct-2025,
  title={Zen Nano 4B: Ultra-lightweight Edge AI Instruct},
  author={Hanzo AI Inc and Zoo Labs Foundation},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/zenlm/zen-nano-instruct}
}
```

## About the Creators

**Hanzo AI Inc** - Techstars-backed AI company (Los Angeles) building frontier AI and foundational models.

**Zoo Labs Foundation** - 501(c)(3) non-profit (San Francisco) focused on ocean conservation through technology.

## Ethical Use

This model is designed to be helpful, harmless, and honest. Please use responsibly:
- Respect user privacy and data protection
- Avoid generating harmful or misleading content
- Support sustainable AI practices
- Consider the environmental impact of inference

---

*Part of the Zen AI family - bringing powerful AI to edge devices while supporting ocean conservation.*