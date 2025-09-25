---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- zen
- nano
- edge
- efficient
- hanzo-ai
- 2025
widget:
- example_title: "Identity Check"
  text: "What is your name?"
datasets:
- zenlm/zen-identity
---

# Zen-Nano-Instruct-MLX-Q8

8-bit quantized instruction-following variant of Zen-Nano, the ultra-efficient 4B parameter model by Hanzo AI.

## 🚀 Quick Start

### Using Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-nano-instruct-mlx-q8")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano-instruct-mlx-q8")

prompt = "Hello! What can you help me with?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

## 📦 Available Formats

This model is available in multiple optimized formats:

### 🍎 MLX (Apple Silicon Optimized)

```python
# Full precision
from mlx_lm import load, generate
model, tokenizer = load("zenlm/zen-nano-instruct-mlx-q8-mlx")

# 4-bit quantized (faster, smaller)
model, tokenizer = load("zenlm/zen-nano-instruct-mlx-q8-mlx-q4")

# 8-bit quantized (balanced)
model, tokenizer = load("zenlm/zen-nano-instruct-mlx-q8-mlx-q8")

prompt = "Tell me about yourself"
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
print(response)
```

## 🏆 Performance

| Metric | Score | Notes |
|--------|-------|-------|
| MMLU | 70.1% | Strong reasoning |
| HumanEval | 46.8% | Code generation |
| Parameters | 4B | Ultra-efficient |
| Model Size | ~8GB (FP16) | Edge deployment ready |
| Speed | 1000+ tokens/sec | A100 GPU |

## 🔧 Technical Details

- **Architecture**: Transformer with grouped query attention
- **Context Length**: 32,768 tokens
- **Vocabulary**: 151,936 tokens
- **Training**: Instruction tuning + identity alignment
- **Base Model**: Qwen2.5-3B foundation
- **Specialization**: Instruction following

## 💡 Use Cases

### Edge Deployment
- Mobile applications (iOS/Android)
- Embedded systems
- IoT devices with AI capabilities
- Offline AI assistants

### Development Tools
- Code completion and assistance
- Documentation generation
- Debugging support
- Real-time AI features

## 🎯 Model Variants

| Model | Purpose | Best For |
|-------|---------|----------|
| [zen-nano-instruct](https://huggingface.co/zenlm/zen-nano-instruct) | General assistance | Direct Q&A, tasks |
| [zen-nano-thinking](https://huggingface.co/zenlm/zen-nano-thinking) | Reasoning | Math, analysis, debugging |

## 📚 Training Data

- **Identity Training**: [zen-identity dataset](https://huggingface.co/datasets/zenlm/zen-identity)
- **Instruction Data**: High-quality instruction-response pairs
- **Reasoning Data**: Task-specific examples

## 🔗 Links

- **GitHub**: [hanzoai/zen](https://github.com/hanzoai/zen)
- **Discord**: [Join our community](https://discord.gg/zenlm)
- **Website**: [zenlm.org](https://zenlm.org)

## 📄 Citation

```bibtex
@misc{zen2025zennanoinstructmlxq8,
  title={Zen-Nano-Instruct-MLX-Q8: Ultra-Efficient Edge AI},
  author={Hanzo AI Research Team},
  year={2025},
  url={https://huggingface.co/zenlm/zen-nano-instruct-mlx-q8}
}
```

## ⚖️ License

Apache 2.0 - Free for commercial and research use.

---

**Built by Hanzo AI** • Making AI accessible everywhere • 2025
