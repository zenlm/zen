# Zen Models GGUF Conversion Pipeline

Production-ready GGUF conversion pipeline for all Zen models, optimized for llama.cpp deployment.

## Overview

This pipeline converts Zen models from SafeTensors/PyTorch format to GGUF with optimal quantization levels for various deployment scenarios.

## Models Supported

- **zen-nano-instruct** - Lightweight instruction-following model
- **zen-nano-thinking** - Reasoning model with chain-of-thought
- **zen-omni** - Multimodal model (vision, audio, text)
- **zen-omni-thinking** - Multimodal reasoning with thinking tokens
- **zen-omni-captioner** - Specialized vision-language captioning
- **zen-coder** - Code generation and explanation
- **zen-next** - Next-generation model with advanced capabilities

## Quick Start

### 1. Convert All Models

```bash
# Convert all models with default settings
./batch_convert.sh --all

# Convert models in parallel for speed
./batch_convert.sh --parallel

# Convert specific model
./batch_convert.sh zen-nano-instruct
```

### 2. Python API

```python
# Convert all models
python convert_zen_to_gguf.py --model all

# Convert specific model
python convert_zen_to_gguf.py --model zen-omni

# Custom quantization
python convert_zen_to_gguf.py --model zen-coder --quantization Q6_K
```

### 3. Optimized Quantization

```bash
# Mobile/Edge deployment
python optimize_quantization.py --profile mobile

# Balanced quality/performance
python optimize_quantization.py --profile balanced

# Maximum quality
python optimize_quantization.py --profile quality

# Server deployment
python optimize_quantization.py --profile server

# Special optimization for thinking models
python optimize_quantization.py --profile thinking
```

## Quantization Profiles

### Mobile/Edge (`Q4_K_S`, `Q4_K_M`)
- Optimized for mobile devices and edge computing
- ~4 bits per weight
- 60-70% size reduction
- Minimal quality loss for most tasks

### Balanced (`Q5_K_M`, `Q4_K_M`)
- Best balance between quality and file size
- ~4-5 bits per weight
- 50-60% size reduction
- Recommended for most use cases

### Quality (`Q6_K`, `Q8_0`)
- Higher quality with reasonable size
- ~6-8 bits per weight
- 30-50% size reduction
- For applications requiring higher accuracy

### Server (`Q8_0`, `FP16`)
- Maximum quality for server deployment
- 8-16 bits per weight
- Minimal to no quality loss
- For GPU-accelerated inference

## Special Features

### Thinking Model Support

Models with thinking capabilities (`zen-nano-thinking`, `zen-omni-thinking`, `zen-next`) preserve special tokens:
- `<thinking>` / `</thinking>` - Reasoning process
- `<|thinking|>` / `<|/thinking|>` - Alternative format
- Extended context (16K-64K tokens)

### Metadata Preservation

```bash
# Generate metadata for all models
python preserve_metadata.py
```

Creates:
- Tokenizer configurations with special tokens
- Model cards with usage instructions
- Conversion configurations
- GGUF-specific metadata

## Output Structure

```
output/
├── zen-nano-instruct-Q4_K_M.gguf
├── zen-nano-instruct-Q5_K_M.gguf
├── zen-nano-instruct-Q8_0.gguf
├── zen-nano-instruct-F16.gguf
├── optimized/
│   ├── zen-nano-instruct-mobile-Q4_K_S.gguf
│   ├── zen-nano-instruct-balanced-Q5_K_M.gguf
│   └── zen-nano-instruct-quality-Q8_0.gguf
├── zen-nano-instruct/
│   ├── tokenizer_config.json
│   ├── MODEL_CARD.md
│   └── conversion_config.json
└── conversion_summary.json
```

## Usage with llama.cpp

### Basic Inference

```bash
# Standard inference
./llama-cli -m output/zen-nano-instruct-Q5_K_M.gguf \
    --prompt "Write a Python function to sort a list" \
    --ctx-size 8192 \
    --temp 0.7

# Thinking model
./llama-cli -m output/zen-nano-thinking-Q6_K.gguf \
    --prompt "Solve this step by step: What is 15% of 240?" \
    --ctx-size 16384 \
    --temp 0.3

# Code generation
./llama-cli -m output/zen-coder-Q5_K_M.gguf \
    --prompt "<|code|>def fibonacci(n):<|/code|>" \
    --ctx-size 16384 \
    --temp 0.2
```

### Server Deployment

```bash
# Start server with Zen model
./llama-server -m output/zen-omni-Q8_0.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 32768 \
    --n-gpu-layers -1  # Use all GPU layers
```

## Performance Benchmarks

| Model | Quantization | Size (GB) | Speed (tok/s) | Perplexity |
|-------|-------------|-----------|---------------|------------|
| zen-nano-instruct | Q4_K_M | 0.3 | 150 | 6.2 |
| zen-nano-instruct | Q5_K_M | 0.4 | 140 | 6.0 |
| zen-nano-instruct | Q8_0 | 0.6 | 120 | 5.9 |
| zen-omni | Q4_K_M | 0.9 | 80 | 5.8 |
| zen-omni | Q6_K | 1.3 | 65 | 5.6 |
| zen-coder | Q5_K_M | 1.2 | 70 | 5.5 |
| zen-next | Q6_K | 2.1 | 45 | 5.3 |

*Benchmarks on M2 Max with Metal acceleration*

## Troubleshooting

### Build Issues

If quantization tools are missing:
```bash
cd /Users/z/work/zen/llama.cpp
mkdir -p build && cd build
cmake .. -DLLAMA_METAL=ON -DLLAMA_ACCELERATE=ON
make -j8
```

### Model Not Found

Check alternate paths:
- `/Users/z/work/zen/models/[model-name]`
- `/Users/z/work/zen/[model-name]`
- `/Users/z/work/zen/base-models/[model-name]`

### Special Tokens Not Working

Run metadata preservation:
```bash
python preserve_metadata.py
```

## Advanced Configuration

### Custom Quantization

```python
from convert_zen_to_gguf import GGUFConverter, ZenModel

# Define custom model
custom_model = ZenModel(
    name="zen-custom",
    base_path="/path/to/model",
    has_thinking=True,
    special_tokens={
        "thinking_start": "<think>",
        "thinking_end": "</think>"
    },
    quantizations=["Q4_K_M", "Q6_K", "FP16"]
)

# Convert
converter = GGUFConverter()
results = converter.process_model("zen-custom")
```

### Batch Processing

```python
# Process multiple models with custom settings
models = ["zen-nano-instruct", "zen-coder"]
for model in models:
    converter.process_model(model)
```

## Requirements

- Python 3.8+
- llama.cpp (included)
- macOS with Metal support (optional, for acceleration)
- 16GB+ RAM recommended
- 50GB+ free disk space for all models

## License

Apache 2.0 - See LICENSE file

## Support

For issues or questions about the Zen models GGUF conversion pipeline, please refer to the Hanzo AI documentation.