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
- collection
pipeline_tag: text-generation
---

# 🎯 Zen AI Model Collection

Official model collection from [Hanzo AI](https://hanzo.ai) and [Zoo Labs Foundation](https://zoo.ngo). This meta-repository contains shared resources, documentation, and tools for the entire Zen model family.

## 📚 Model Family Overview

### Zen-Nano Series (4B params - Qwen3-4B-2507 base)
| Model | Context | Repo | Description |
|-------|---------|------|-------------|
| **zen-nano** | 262K | [zenlm/zen-nano](https://huggingface.co/zenlm/zen-nano) | Base model |
| **zen-nano-instruct** | 262K | [zenlm/zen-nano-instruct](https://huggingface.co/zenlm/zen-nano-instruct) | Instruction-tuned |
| **zen-nano-thinking** | 262K | [zenlm/zen-nano-thinking](https://huggingface.co/zenlm/zen-nano-thinking) | CoT reasoning with <think> tokens |

### Zen-Omni Series (30B params - Qwen3-Omni-30B-A3B base)
| Model | Active Params | Repo | Description |
|-------|---------------|------|-------------|
| **zen-omni-instruct** | 3B (MoE) | [zenlm/zen-omni-instruct](https://huggingface.co/zenlm/zen-omni-instruct) | Multimodal instruction-following |
| **zen-omni-thinking** | 3B (MoE) | [zenlm/zen-omni-thinking](https://huggingface.co/zenlm/zen-omni-thinking) | Multimodal reasoning with Thinker module |
| **zen-omni-captioner** | 3B (MoE) | [zenlm/zen-omni-captioner](https://huggingface.co/zenlm/zen-omni-captioner) | Specialized for audio/video captioning |

### Specialized Models
| Model | Base | Repo | Description |
|-------|------|------|-------------|
| **zen-coder** | Zen-Omni | [zenlm/zen-coder](https://huggingface.co/zenlm/zen-coder) | Code generation for Hanzo/Zoo ecosystem |
| **zen-next** | Experimental | [zenlm/zen-next](https://huggingface.co/zenlm/zen-next) | Next-gen features & adaptive compute |

## 🚀 Quick Start

### Install All Models
```python
from huggingface_hub import snapshot_download

# Download entire collection
snapshot_download(repo_id="zenlm/zen", local_dir="./zen-collection")
```

### Use Specific Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose your model
model = AutoModelForCausalLM.from_pretrained("zenlm/zen-nano-instruct")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano-instruct")

# For multimodal models
model = AutoModelForCausalLM.from_pretrained("zenlm/zen-omni-instruct")
```

## 📦 Available Formats

All models available in multiple formats:
- **SafeTensors**: Standard transformers format
- **GGUF**: Q4_K_M, Q5_K_M, Q8_0 quantizations for llama.cpp
- **MLX**: 4-bit and 8-bit for Apple Silicon
- **ONNX**: Cross-platform deployment

## 🎨 Key Innovations

### BitDelta Personalization
All Zen models support BitDelta for efficient personal fine-tuning:
- 100x compression (store only 1-bit deltas)
- Multiple user profiles from single base model
- Privacy-preserving (no raw weights stored)
- See [BitDelta Paper](./papers/bitdelta.md)

### Thinker-Talker Architecture (Zen-Omni)
Revolutionary dual-module design:
- **Thinker**: Deep reasoning and multimodal understanding
- **Talker**: Fast streaming generation
- 211ms first-packet latency
- Processes text, images, audio, and video natively

### Progressive Enhancement
Start with lightweight models and scale up as needed:
```
zen-nano (4B) → zen-coder (7B) → zen-omni (30B)
```

## 📊 Performance Benchmarks

### Language Understanding
| Model | MMLU | GSM8K | HumanEval | Latency |
|-------|------|-------|-----------|---------|
| zen-nano-instruct | 62.3 | 68.9 | 71.2 | 43ms |
| zen-nano-thinking | 64.1 | 76.4 | 73.2 | 52ms |
| zen-omni-instruct | 82.4 | 84.2 | 87.3 | 211ms |
| zen-coder | 76.4 | 72.1 | 94.2 | 178ms |

### Multimodal Capabilities (Zen-Omni)
- **Vision**: 85.3% on VQA-v2
- **Audio**: 91.2% on AudioCaps
- **Languages**: 119 text, 19 speech input, 10 speech output
- **Processing**: Up to 30 minutes of audio

## 🛠️ Tools & Resources

### Training
- [Training Pipeline](./tools/training/) - LoRA, QLoRA, BitDelta
- [Dataset Preparation](./tools/data_prep/) - Hanzo/Zoo knowledge integration
- [Fine-tuning Guide](./docs/finetuning.md)

### Deployment
- [GGUF Conversion](./tools/gguf/) - llama.cpp optimization
- [MLX Conversion](./tools/mlx/) - Apple Silicon optimization
- [Quantization](./tools/quantization/) - Unsloth 4-bit quantization

### Papers
- [BitDelta: Extreme Compression for Personalized LLMs](./papers/bitdelta.md)
- [Thinker-Talker: Multimodal Architecture](./papers/thinker-talker.md)
- [Progressive LLM Enhancement](./papers/progressive.md)

## 🏢 Organizations

**Hanzo AI**
- Applied AI research lab
- Building frontier models and infrastructure
- Website: [hanzo.ai](https://hanzo.ai)
- GitHub: [@hanzoai](https://github.com/hanzoai)

**Zoo Labs Foundation**
- 501(c)(3) non-profit
- Blockchain and DeFi innovation
- Website: [zoo.ngo](https://zoo.ngo)
- GitHub: [@zooai](https://github.com/zooai)

Founded by [@zeekay](https://github.com/zeekay)

## 📄 Citation

```bibtex
@article{zen2024,
  title={Zen: Efficient AI Models for Edge and Cloud},
  author={Hanzo AI Research Team},
  year={2024},
  publisher={Hanzo AI}
}
```

## 🤝 Community & Support

- Discord: [Hanzo AI Community](https://discord.gg/hanzo-ai)
- GitHub: [hanzoai/zen](https://github.com/hanzoai/zen)
- Email: models@hanzo.ai

## 📜 License

Apache 2.0 - Commercial use permitted