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
- 4bit
- quantized
base_model: Qwen/Qwen3-4B-Instruct
datasets:
- zenlm/zen-identity-v1
library_name: mlx
pipeline_tag: text-generation
---

# Zen Nano 4B Instruct (4-bit Quantized)

🌊 **Ultra-efficient edge AI assistant** with 4-bit quantization for maximum performance and minimal memory usage.

Jointly developed by [Hanzo AI Inc](https://hanzo.ai) (Techstars-backed, Los Angeles) and [Zoo Labs Foundation](https://zoolabs.org) (501c3, San Francisco).

## Model Details

- **Architecture**: Qwen3 4B Instruct
- **Parameters**: 4 billion (4-bit quantized)
- **Context Length**: 32,768 tokens
- **Vocabulary Size**: 151,936 tokens
- **Size**: ~2.5GB (compressed from ~8GB)
- **Training**: Fine-tuned for identity alignment and instruction following
- **License**: Apache 2.0

## Features

⚡ **Ultra-efficient**: 4-bit quantization for fastest inference
💾 **Memory optimized**: ~2.5GB RAM usage vs 8GB for full model
🍎 **Apple Silicon optimized**: MLX format for M1/M2/M3/M4 Macs
📱 **Mobile-ready**: Efficient enough for mobile and edge devices
🛡️ **Privacy-first**: Runs entirely on-device
🌍 **Mission-aligned**: Part of our ocean conservation initiative

## Available Formats

### MLX Format (Apple Silicon - M1/M2/M3/M4)
- **4-bit Quantized**: ~2.5GB - This ultra-efficient version
- **Full Precision**: Available in [zen-nano-instruct](https://huggingface.co/zenlm/zen-nano-instruct)

### GGUF Format (Universal - llama.cpp)
- **Q4_K_M**: ~2.5GB - 4-bit medium quality (recommended)
- **Q5_K_M**: ~3.1GB - 5-bit medium quality
- **Q6_K**: ~3.7GB - 6-bit quality
- **Q8_0**: ~4.3GB - 8-bit quality

## Usage

### MLX (Apple Silicon)

```python
from mlx_lm import load, generate

# Load 4-bit quantized model
model, tokenizer = load("zenlm/zen-nano-instruct-4bit")

# Generate response
response = generate(
    model,
    tokenizer,
    prompt="User: Explain renewable energy in simple terms\nAssistant:",
    max_tokens=150,
    temp=0.7
)
print(response)
```

### GGUF (Universal - llama.cpp)

```bash
# Download GGUF version
wget https://huggingface.co/zenlm/zen-nano-instruct-4bit/resolve/main/zen-nano-instruct-4bit-Q4_K_M.gguf

# Run with llama.cpp
./llama-cli -m zen-nano-instruct-4bit-Q4_K_M.gguf -p "User: What is climate change?\nAssistant:" -n 100

# Interactive chat
./llama-cli -m zen-nano-instruct-4bit-Q4_K_M.gguf --interactive-first --reverse-prompt "User:"
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

# Load GGUF model
llm = Llama(
    model_path="zen-nano-instruct-4bit-Q4_K_M.gguf",
    n_ctx=2048,
    verbose=False
)

# Generate
output = llm(
    prompt="User: How can I reduce my carbon footprint?\nAssistant:",
    max_tokens=100,
    stop=["User:", "\n\n"]
)

print(output['choices'][0]['text'])
```

### Ollama

```bash
# Create Modelfile
echo 'FROM ./zen-nano-instruct-4bit-Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER stop "User:"
SYSTEM "You are Zen Nano, an AI assistant created by Hanzo AI Inc and Zoo Labs Foundation to democratize AI while protecting our oceans."' > Modelfile

# Import model
ollama create zen-nano-4bit -f Modelfile

# Run
ollama run zen-nano-4bit "Tell me about ocean conservation"
```

## Quality vs Efficiency

The 4-bit quantization provides excellent quality retention while dramatically reducing resource requirements:

- **Quality**: ~95% of full-precision performance
- **Speed**: 2-3x faster inference on most hardware
- **Memory**: ~70% reduction in RAM usage
- **Storage**: ~70% smaller download size

## Identity & Mission

- **Name**: Zen Nano 4B (4-bit Quantized)
- **Creators**: Hanzo AI Inc & Zoo Labs Foundation
- **Mission**: Democratizing AI while protecting our oceans
- **Values**: Privacy-first, sustainable AI, accessible technology

## Model Family

This is part of the complete Zen Nano ecosystem:

- **[zen-nano-instruct](https://huggingface.co/zenlm/zen-nano-instruct)**: Full-precision instruction following
- **[zen-nano-instruct-4bit](https://huggingface.co/zenlm/zen-nano-instruct-4bit)**: This ultra-efficient quantized model
- **[zen-nano-thinking](https://huggingface.co/zenlm/zen-nano-thinking)**: Advanced reasoning with step-by-step thinking
- **[zen-nano-thinking-4bit](https://huggingface.co/zenlm/zen-nano-thinking-4bit)**: Ultra-efficient reasoning model

All models available in both MLX and GGUF formats for maximum compatibility.

## Training Details

Based on Qwen3-4B-Instruct, fine-tuned using MLX LoRA with specialized datasets for:
- Proper identity and attribution
- Ocean conservation mission alignment  
- Privacy-first principles
- Helpful and harmless responses
- Optimized for 4-bit quantization performance

## Performance Benchmarks

### MLX (Apple M1 Pro)
- **Inference Speed**: 60-120 tokens/second (2x faster than full model)
- **Memory Usage**: ~2.5GB RAM
- **Context Processing**: ~800 tokens/second
- **Quality Score**: 95% vs full precision

### GGUF (Apple M1 Pro, CPU-only)
| Quantization | Size | Tokens/sec | Quality |
|--------------|------|------------|---------|
| Q4_K_M       | 2.5GB| 20-30      | 95%     |
| Q5_K_M       | 3.1GB| 15-25      | 97%     |
| Q8_0         | 4.3GB| 10-20      | 99%     |

## Use Cases

Perfect for:
- **Mobile applications**: Lightweight enough for smartphones
- **Edge computing**: IoT devices and embedded systems
- **Privacy-sensitive applications**: No data leaves your device
- **Educational tools**: Fast, accessible AI tutoring
- **Rapid prototyping**: Quick iteration with minimal resources
- **Battery-powered devices**: Efficient inference extends battery life

## License

Apache 2.0 - Free for commercial and research use.

## Citation

```bibtex
@model{zen-nano-instruct-4bit-2025,
  title={Zen Nano 4B Instruct: Ultra-efficient Edge AI with 4-bit Quantization},
  author={Hanzo AI Inc and Zoo Labs Foundation},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/zenlm/zen-nano-instruct-4bit}
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

*Ultra-efficient AI for everyone - bringing advanced capabilities to edge devices while supporting ocean conservation.*