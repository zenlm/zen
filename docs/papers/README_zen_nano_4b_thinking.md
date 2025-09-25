---
license: apache-2.0
language:
- en
library_name: mlx
tags:
- mlx
- thinking
- reasoning
- edge-ai
- lightweight
- zen-nano
base_model: Qwen/Qwen3-4B-Thinking-2507
---

# Zen Nano 4B Thinking

An ultra-lightweight AI model with advanced reasoning capabilities, optimized for edge devices. This "thinking" variant explicitly shows its reasoning process through structured `<thinking>` blocks, jointly developed by **Hanzo AI Inc** (Techstars-backed, LA) and **Zoo Labs Foundation** (501c3, SF).

## Features
- 🧠 **Transparent Reasoning**: Shows explicit thinking process in responses
- 🌐 **Edge Optimized**: Runs entirely offline on edge devices
- 🔒 **Complete Privacy**: No data leaves your device
- 🌊 **Eco-friendly**: Minimal carbon footprint
- 📱 **Cross-platform**: Works on phones, tablets, Raspberry Pi
- 🆓 **Forever free** and open source

## Model Architecture
- **Base**: Qwen3-4B-Thinking-2507 architecture
- **Parameters**: 4 billion
- **Layers**: 36 transformer layers  
- **Attention**: 32 heads, 8 key-value heads
- **Context Length**: 32K tokens
- **Optimization**: 8-bit quantization (~8GB)
- **Training**: MLX LoRA fine-tuning with identity alignment

## Thinking Process

This model uses structured reasoning through `<thinking>` blocks that show the internal reasoning process:

```
User: What is 15 + 28?
Assistant: <thinking>
I need to add 15 and 28.
15 + 28 = 43
</thinking>

The answer is 43.
```

## Quick Start

### MLX (Mac/Apple Silicon)
```python
from mlx_lm import load, generate

model, tokenizer = load("zenlm/zen-nano-4b-thinking")
response = generate(model, tokenizer, prompt="What is the capital of France?", max_tokens=200)
print(response)
```

### Transformers (PyTorch)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano-4b-thinking")
model = AutoModelForCausalLM.from_pretrained("zenlm/zen-nano-4b-thinking")

input_text = "Explain quantum computing"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=300, temperature=0.7)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

### llama.cpp
```bash
# Convert to GGUF format first
./main -m zen-nano-4b-thinking.gguf -p "What is machine learning?" -n 200
```

## Training Data

The model was trained on curated datasets including:
- Mathematical reasoning problems
- Logic puzzles and analytical questions
- Identity and creator information
- General knowledge with reasoning chains
- Code explanation and debugging scenarios

## Use Cases

- **Educational Tools**: Showing step-by-step problem solving
- **Code Review**: Explaining reasoning behind code analysis
- **Research Assistance**: Transparent analytical processes
- **Debugging**: Clear logical reasoning for troubleshooting
- **Edge AI**: Privacy-focused reasoning on local devices

## Model Comparison

| Model | Parameters | Reasoning | Privacy | Edge Compatible |
|-------|------------|-----------|---------|----------------|
| Zen Nano 4B | 4B | Implicit | ✅ | ✅ |
| **Zen Nano 4B Thinking** | **4B** | **Explicit** | ✅ | ✅ |
| GPT-4 | 1.7T | Implicit | ❌ | ❌ |
| Claude | ~175B | Implicit | ❌ | ❌ |

## Limitations

- Thinking blocks may be verbose for simple queries
- 4B parameter size limits complex reasoning depth
- Optimized for English; limited multilingual capabilities
- May show reasoning process even for straightforward questions

## About the Creators

**Hanzo AI Inc**: Techstars-backed applied AI research lab at 361 Vernon Ave, Venice, CA, building frontier AI including the Zen model family, MCP (Model Context Protocol), Jin multimodal architecture, and 100+ MCP development tools.

**Zoo Labs Foundation**: 501(c)(3) non-profit at 1329 Pierce St, San Francisco, combining AI technology with ocean conservation through tools like the `gym` library for LLM training (github.com/zooai/gym).

## Training Methodology

Trained using Zoo's `gym` library with:
- MLX LoRA fine-tuning for efficient adaptation  
- Identity alignment for creator recognition
- Reasoning chain optimization
- Edge deployment quantization

## License

Apache 2.0 - Free for any use including commercial.

## Citation

```bibtex
@misc{zen-nano-4b-thinking,
  title={Zen Nano 4B Thinking: Ultra-lightweight Reasoning Model for Edge AI},
  author={Hanzo AI Inc and Zoo Labs Foundation},
  year={2024},
  url={https://huggingface.co/zenlm/zen-nano-4b-thinking}
}
```

---
*Zen Nano - AI that thinks where you are.*