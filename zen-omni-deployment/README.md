# Zen-Omni Deployment Package

Complete deployment package for Zen-Omni multimodal foundation models with Thinker-Talker architecture.

## Overview

Zen-Omni is a 30B parameter multimodal foundation model with only 3B active parameters through Mixture of Experts (MoE), achieving ultra-low latency (211ms) while maintaining state-of-the-art performance across text, image, audio, and video understanding tasks.

## Model Variants

### 1. Zen-Omni-Thinking
- **Focus**: Deep reasoning and complex problem-solving
- **Weight Distribution**: 70% Thinker, 30% Talker
- **Use Cases**: Mathematical reasoning, code generation, scientific analysis
- **HF Hub**: `zenlm/zen-omni-thinking`

### 2. Zen-Omni-Talking
- **Focus**: Fast, fluent generation with streaming support
- **Weight Distribution**: 30% Thinker, 70% Talker
- **Use Cases**: Conversational AI, real-time translation, interactive assistants
- **HF Hub**: `zenlm/zen-omni-talking`
- **Latency**: 185ms first token

### 3. Zen-Omni-Captioner
- **Focus**: Audio and video understanding with temporal alignment
- **Weight Distribution**: 50% Thinker, 50% Talker
- **Use Cases**: Video captioning, audio transcription, multimodal summarization
- **HF Hub**: `zenlm/zen-omni-captioner`

## Architecture Highlights

- **Thinker-Talker Design**: Dual-module architecture separating reasoning from generation
- **Mixture of Experts**: 10 experts with top-2 routing (3B/30B active parameters)
- **Multimodal Processing**: Unified encoders for text, image, audio, and video
- **Streaming Support**: Real-time generation with sub-200ms latency

## Directory Structure

```
zen-omni-deployment/
├── paper/
│   ├── zen-omni.tex         # LaTeX paper
│   └── Makefile              # Build script for paper
├── models/
│   ├── zen-omni-thinking/   # Thinking variant
│   │   ├── README.md         # Model card
│   │   ├── config.json       # Model configuration
│   │   └── ...               # Model files
│   ├── zen-omni-talking/    # Talking variant
│   │   └── ...
│   └── zen-omni-captioner/  # Captioner variant
│       └── ...
├── deploy_to_hf.py           # Deployment script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Deploy to Hugging Face

Deploy a specific variant:
```bash
python deploy_to_hf.py --variant thinking
python deploy_to_hf.py --variant talking
python deploy_to_hf.py --variant captioner
```

Deploy all variants:
```bash
python deploy_to_hf.py --all
```

### Build Paper

```bash
cd paper
make all
make view  # Open PDF
```

## Model Usage

### Basic Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model (choose variant)
model = AutoModelForCausalLM.from_pretrained(
    "zenlm/zen-omni-thinking",  # or talking/captioner
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-omni-thinking")

# Generate response
prompt = "Explain quantum computing in simple terms"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Streaming Generation (Talking Variant)

```python
from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_special_tokens=True)
model.generate(
    **inputs,
    max_new_tokens=200,
    streamer=streamer
)
```

### Multimodal Processing

```python
from PIL import Image

# Load image
image = Image.open("example.jpg")

# Process with model
inputs = tokenizer(
    "Describe this image:",
    images=image,
    return_tensors="pt"
)
outputs = model.generate(**inputs)
```

## Performance Benchmarks

| Model | MMLU | VQA | Latency | Tokens/sec |
|-------|------|-----|---------|------------|
| Zen-Omni-Thinking | 87.2% | 88.9% | 280ms | 55 |
| Zen-Omni-Talking | 84.5% | 87.3% | 185ms | 75 |
| Zen-Omni-Captioner | 83.9% | 86.8% | 211ms | 65 |

## Technical Specifications

- **Total Parameters**: 30B
- **Active Parameters**: 3B (MoE with 10 experts, top-2)
- **Context Length**: 16K-32K tokens (variant dependent)
- **Modalities**: Text, Image (336x336), Audio (16kHz), Video (30 FPS)
- **Precision**: FP16/INT8 quantization supported
- **Memory**: 12GB (FP16 active), 60GB (full model)

## API Endpoints

Models will be available through the Zen API:

```python
import requests

response = requests.post(
    "https://api.zenlm.ai/v1/chat/completions",
    headers={"Authorization": "Bearer YOUR_KEY"},
    json={
        "model": "zen-omni-talking",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True
    }
)
```

## Citation

```bibtex
@article{zen2025omni,
  title={Zen-Omni: A Thinker-Talker Architecture for Ultra-Low Latency Multimodal Understanding},
  author={Zen Research Team},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## License

Apache 2.0

## Contact

- Research: research@zenlm.ai
- Support: support@zenlm.ai
- GitHub: https://github.com/zenlm/zen-omni

## Acknowledgments

Based on the Qwen3-Omni architecture with novel Thinker-Talker design innovations. Thanks to the open-source community for datasets and evaluation benchmarks.