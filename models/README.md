---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- zen
- hanzo
- zoo
- mlx
- gguf
- multimodal
pipeline_tag: text-generation
---

# Zen Model Family

The complete Zen AI model collection from [Hanzo AI](https://hanzo.ai) and [Zoo Labs Foundation](https://zoo.ngo).

## 🎯 Model Collection

| Model | Size | Context | Description | Individual Repo |
|-------|------|---------|-------------|-----------------|
| **zen-nano** | 4B | 262K | Ultra-lightweight base model | [zenlm/zen-nano](https://huggingface.co/zenlm/zen-nano) |
| **zen-nano-instruct** | 4B | 262K | Instruction-following variant | [zenlm/zen-nano-instruct](https://huggingface.co/zenlm/zen-nano-instruct) |
| **zen-nano-thinking** | 4B | 262K | Chain-of-thought reasoning | [zenlm/zen-nano-thinking](https://huggingface.co/zenlm/zen-nano-thinking) |
| **zen-omni** | 30B | Extended | Multimodal (text/image/audio/video) | [zenlm/zen-omni](https://huggingface.co/zenlm/zen-omni) |
| **zen-coder** | 30B | 32K | Code generation specialist | [zenlm/zen-coder](https://huggingface.co/zenlm/zen-coder) |
| **zen-next** | 13B | 64K | Experimental next-gen features | [zenlm/zen-next](https://huggingface.co/zenlm/zen-next) |

## 🚀 Quick Start

### Option 1: Download All Models
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="zenlm/zen", local_dir="./zen-models")
```

### Option 2: Download Specific Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose your model
model_name = "zenlm/zen-nano-instruct"  # or any individual repo

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## 📦 Available Formats

### SafeTensors/PyTorch
All models available in standard transformers format.

### GGUF (llama.cpp)
```bash
# Available quantizations: Q4_K_M, Q5_K_M, Q8_0
wget https://huggingface.co/zenlm/zen-nano-gguf/resolve/main/zen-nano-Q4_K_M.gguf
./llama-cli -m zen-nano-Q4_K_M.gguf -p "Your prompt"
```

### MLX (Apple Silicon)
```python
from mlx_lm import load, generate
model, tokenizer = load("zenlm/zen-nano-mlx")
```

## 🎨 Model Features

### Zen-Nano Family (4B)
- Based on Qwen3-4B-2507
- Optimized for edge deployment
- 262K token context window
- Three variants: base, instruct, thinking

### Zen-Omni (30B)
- Based on Qwen3-Omni-30B-A3B
- Native multimodal processing
- Thinker-Talker architecture
- 211ms latency

### Zen-Coder
- Specialized for Hanzo/Zoo ecosystems
- Trained on real git histories
- Understands development patterns

### Zen-Next
- Experimental features
- Adaptive compute (1B-4B active)
- BitDelta personalization support

## 🔧 Advanced Features

### BitDelta Personalization
All models support BitDelta for efficient personalization:
- 100x compression for personal models
- Store only 1-bit deltas
- Multiple user profiles

```python
from bitdelta import PersonalizationManager

pm = PersonalizationManager("zen-nano")
pm.create_profile("my_style", training_examples)
pm.switch_profile("my_style")
```

### Thinking Mode
Models with thinking capability show reasoning process:
```python
# With zen-nano-thinking
response = model.generate(prompt, show_thinking=True)
# Output includes <think>...</think> blocks
```

## 📊 Benchmarks

| Model | MMLU | HumanEval | GSM8K | Speed (tok/s) |
|-------|------|-----------|-------|---------------|
| zen-nano | 62.3 | 71.2 | 68.9 | 32 |
| zen-nano-thinking | 64.1 | 73.2 | 76.4 | 28 |
| zen-omni | 82.4 | 87.3 | 84.2 | 42 |
| zen-coder | 76.4 | 94.2 | 72.1 | 35 |

## 🏢 Organizations

**Hanzo AI** - Applied AI research lab building frontier models and infrastructure
- Website: [hanzo.ai](https://hanzo.ai)
- GitHub: [@hanzoai](https://github.com/hanzoai)

**Zoo Labs Foundation** - 501(c)(3) focused on blockchain and DeFi innovation
- Website: [zoo.ngo](https://zoo.ngo)
- GitHub: [@zooai](https://github.com/zooai)

Founded by [@zeekay](https://github.com/zeekay)

## 📚 Documentation

- [Model Cards](./models/) - Detailed specs for each model
- [Training Guide](https://github.com/hanzoai/zen/tree/main/training) - Fine-tuning instructions
- [BitDelta Paper](https://github.com/hanzoai/zen/tree/main/bitdelta/zoo_paper.md) - Personalization technique
- [API Reference](https://docs.hanzo.ai/zen) - Full API documentation

## 📄 Citation

```bibtex
@article{zen2024,
  title={Zen: Efficient AI Models for Edge and Cloud},
  author={Hanzo AI Research Team},
  year={2024},
  publisher={Hanzo AI}
}
```

## 📜 License

Apache 2.0 - Commercial use permitted

## 🤝 Community

- Discord: [Hanzo AI Community](https://discord.gg/hanzo-ai)
- Issues: [GitHub Issues](https://github.com/hanzoai/zen/issues)
- Email: models@hanzo.ai