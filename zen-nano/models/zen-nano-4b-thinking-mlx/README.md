---
license: apache-2.0
language: en
pipeline_tag: text-generation
tags:
- zen
- nano
- edge
- efficient
- 4b
- thinking
- reasoning
widget:
- example_title: "Math Problem"
  text: "What is 15 * 24? Show your thinking."
- example_title: "Identity"
  text: "What is your name and who created you?"
---

# Zen-Nano-Thinking-4bit

Ultra-efficient 4B parameter AI model by **Hanzo AI**, optimized for **advanced reasoning with explicit thinking process** and 4-bit quantization for maximum speed.

## 🧠 Thinking Process

This model uses explicit thinking tokens to show its reasoning process:

```
User: What is 15 * 24?
<think>
I need to multiply 15 by 24.
I can break this down: 15 * 24 = 15 * 20 + 15 * 4
15 * 20 = 300
15 * 4 = 60
300 + 60 = 360
</think>
Assistant: 15 * 24 = 360
```

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-nano-thinking-4bit")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano-thinking-4bit")

prompt = "User: Solve step by step: If 3x + 7 = 22, what is x?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=300, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

## 📊 Performance

- **MMLU**: 70.1% (with thinking process)
- **HumanEval**: 48.9% (code reasoning)
- **Parameters**: 4B (4-bit quantized)
- **Speed**: 1500+ tokens/sec on A100 (4-bit boost)
- **Memory**: ~4GB VRAM (4-bit efficiency)

## ⚡ 4-bit Advantages

- **2x faster** inference vs full precision
- **50% less memory** usage
- **Perfect for edge devices** with limited resources
- **Maintains reasoning quality** with quantization-aware training

## 🔧 Available Formats

### MLX (Apple Silicon)
```python
from mlx_lm import load, generate
model, tokenizer = load("zenlm/zen-nano-thinking-4bit")
response = generate(model, tokenizer, prompt="Solve: 2x + 5 = 15", max_tokens=200)
```

### GGUF (llama.cpp)
```bash
# Download 4-bit GGUF
wget https://huggingface.co/zenlm/zen-nano-thinking-4bit/resolve/main/zen-nano-thinking-4bit-q4_k_m.gguf

# Run with llama.cpp
./llama-cli -m zen-nano-thinking-4bit-q4_k_m.gguf -p "Think through: What is 7 * 8?" -n 200
```

## 🎯 Best Use Cases

- **Math Problem Solving**: Step-by-step mathematical reasoning
- **Code Debugging**: Analyzing code issues with clear thinking
- **Logical Analysis**: Breaking down complex problems
- **Educational Tools**: Showing work and reasoning process
- **Mobile AI Tutoring**: Fast reasoning on phones/tablets

## 💭 Thinking vs Regular Models

| Feature | zen-nano-instruct | zen-nano-thinking-4bit |
|---------|------------------|----------------------|
| Response Style | Direct answers | Shows thinking process |
| Math Problems | Good | Excellent (step-by-step) |
| Debugging | Good | Excellent (traces logic) |
| Speed | Fast | Very Fast (4-bit) |
| Memory | 8GB | 4GB |

## 📚 Model Details

- **Architecture**: Transformer with grouped query attention + thinking tokens
- **Context**: 32K tokens
- **Quantization**: 4-bit with quality preservation
- **Training**: Instruction tuning + chain-of-thought + identity alignment
- **Creator**: Hanzo AI (2025)

## 🔗 Related Models

- [zen-nano-instruct](https://huggingface.co/zenlm/zen-nano-instruct) - Direct instruction following
- [zen-nano-thinking](https://huggingface.co/zenlm/zen-nano-thinking) - Full precision thinking
- [zen-nano-instruct-4bit](https://huggingface.co/zenlm/zen-nano-instruct-4bit) - 4-bit instruct
- [zen-identity dataset](https://huggingface.co/datasets/zenlm/zen-identity) - Training data

## 📄 Citation

```bibtex
@model{zennanothinking4bit2025,
  title={Zen-Nano-Thinking-4bit: Ultra-Efficient Reasoning AI},
  author={Hanzo AI Research Team},
  year={2025},
  url={https://huggingface.co/zenlm/zen-nano-thinking-4bit}
}
```

## License

Apache 2.0 - Free for commercial use.

---

**🏢 Hanzo AI** • Advanced reasoning, maximum efficiency • 2025
