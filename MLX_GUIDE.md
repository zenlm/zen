# 🍎 Zen Models - Complete MLX Guide

## Overview
MLX is Apple's framework for efficient machine learning on Apple Silicon. All Zen models are optimized for MLX, providing blazing-fast inference on M1/M2/M3/M4 Macs.

## 🚀 Quick Start - Running Zen Models with MLX

### Installation

```bash
# Install MLX and MLX-LM
pip install mlx mlx-lm

# Install from source for latest features
git clone https://github.com/ml-explore/mlx
cd mlx && pip install -e .
```

### Running Inference

```python
from mlx_lm import load, generate

# Load Zen-Nano-Instruct (already in MLX format!)
model, tokenizer = load("zenlm/zen-nano-instruct")

# Generate response
prompt = "Explain quantum computing in simple terms"
response = generate(model, tokenizer, prompt=prompt, max_tokens=500)
print(response)
```

### Command Line Interface

```bash
# Direct inference with Zen models
mlx_lm.generate \
  --model zenlm/zen-nano-instruct \
  --prompt "Write a haiku about AI" \
  --max-tokens 100

# Interactive chat
mlx_lm.chat \
  --model zenlm/zen-nano-thinking \
  --max-tokens 500
```

## 🏋️ Training with MLX

### Fine-tuning Existing Zen Models

```python
# train_zen_mlx.py
from mlx_lm import load, train, save

# Load existing Zen model
model, tokenizer = load("zenlm/zen-nano-instruct")

# Prepare training data
data = [
    {"prompt": "What's your purpose?", 
     "completion": "I'm Zen-Nano, designed to run efficiently on your device while protecting your privacy."},
    {"prompt": "Who created you?",
     "completion": "I was created by Hanzo AI and Zoo Labs Foundation to democratize AI."}
]

# Training configuration
config = {
    "learning_rate": 1e-5,
    "batch_size": 1,
    "num_epochs": 3,
    "grad_accumulation_steps": 4,
    "warmup_steps": 100,
}

# Fine-tune
trained_model = train(model, tokenizer, data, **config)

# Save the fine-tuned model
save(trained_model, tokenizer, "models/zen-nano-custom-mlx")
```

### LoRA Fine-tuning with MLX

```bash
# Fine-tune Zen models with LoRA
python -m mlx_lm.lora \
  --model zenlm/zen-nano-instruct \
  --train \
  --data ./data/zen_custom.jsonl \
  --batch-size 2 \
  --lora-layers 8 \
  --iters 1000 \
  --learning-rate 1e-5 \
  --adapter-path ./adapters/zen-nano-custom

# Fuse LoRA weights back into model
python -m mlx_lm.fuse \
  --model zenlm/zen-nano-instruct \
  --adapter-path ./adapters/zen-nano-custom \
  --save-path ./models/zen-nano-custom-fused
```

### Training Data Format for MLX

```jsonl
{"text": "User: What is AI?\nAssistant: AI stands for Artificial Intelligence..."}
{"text": "User: Explain machine learning\nAssistant: Machine learning is..."}
```

## 🔧 Advanced MLX Features

### 4-bit Quantization

```python
from mlx_lm import load, quantize

# Load Zen model
model, tokenizer = load("zenlm/zen-nano-instruct")

# Quantize to 4-bit
quantized_model = quantize(model, bits=4, group_size=64)

# Save quantized model
mlx_lm.save(quantized_model, tokenizer, "zen-nano-4bit-mlx")
```

### Memory-Efficient Generation

```python
from mlx_lm import load, generate
import mlx.core as mx

# Load with memory mapping
model, tokenizer = load(
    "zenlm/zen-nano-instruct",
    lazy=True  # Memory-mapped loading
)

# Generate with controlled memory
mx.metal.set_memory_limit(4 * 1024**3)  # 4GB limit
response = generate(
    model, tokenizer,
    prompt="Your question",
    max_tokens=500,
    temp=0.7,
    top_p=0.9
)
```

### Streaming Generation

```python
from mlx_lm import load, stream_generate

model, tokenizer = load("zenlm/zen-nano-thinking")

# Stream tokens as they're generated
for token in stream_generate(model, tokenizer, "Solve this step by step:"):
    print(token, end='', flush=True)
```

## 📦 Converting Between Formats

### Zen Model → MLX Format

```bash
# Our models are already in MLX format, but if needed:
python -m mlx_lm.convert \
  --hf-model zenlm/zen-nano-instruct \
  --output-dir ./mlx-models/zen-nano-instruct \
  --quantize  # Optional 4-bit quantization
```

### MLX → GGUF (for llama.cpp)

```python
# convert_mlx_to_gguf.py
import numpy as np
from mlx_lm import load
import gguf

# Load MLX model
model, tokenizer = load("zenlm/zen-nano-instruct")

# Convert to GGUF
writer = gguf.GGUFWriter("zen-nano.gguf", "zen-nano")

# Add model architecture
writer.add_architecture("llama")  # Zen uses llama architecture
writer.add_context_length(8192)
writer.add_embedding_length(4096)
writer.add_layer_count(32)

# Convert weights
for name, weight in model.items():
    tensor = np.array(weight)
    writer.add_tensor(name, tensor)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
```

## 🎯 Performance Optimization

### Metal Performance Shaders

```python
import mlx.core as mx

# Enable Metal optimizations
mx.metal.init()

# Check Metal availability
print(f"Metal available: {mx.metal.is_available()}")
print(f"Metal device: {mx.metal.get_active_device()}")

# Optimize for specific chip
if "M2" in mx.metal.get_active_device():
    mx.metal.set_cache_limit(8 * 1024**3)  # 8GB for M2
```

### Batch Inference

```python
from mlx_lm import load, generate_batch

model, tokenizer = load("zenlm/zen-nano-instruct")

prompts = [
    "Explain AI",
    "What is machine learning?",
    "Define neural networks"
]

# Batch generation for efficiency
responses = generate_batch(
    model, tokenizer,
    prompts=prompts,
    max_tokens=200,
    batch_size=3
)
```

## 📊 Benchmarking

```python
# benchmark_zen_mlx.py
import time
from mlx_lm import load, generate

model, tokenizer = load("zenlm/zen-nano-instruct")

# Warmup
_ = generate(model, tokenizer, "Test", max_tokens=10)

# Benchmark
prompt = "Write a detailed explanation of quantum computing"
start = time.time()
response = generate(model, tokenizer, prompt, max_tokens=500)
elapsed = time.time() - start

tokens = len(tokenizer.encode(response))
print(f"Tokens generated: {tokens}")
print(f"Time: {elapsed:.2f}s")
print(f"Tokens/sec: {tokens/elapsed:.1f}")
```

## 🔗 MLX Resources

- **MLX GitHub**: [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)
- **MLX Examples**: [github.com/ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)
- **Zen MLX Models**: [huggingface.co/zenlm](https://huggingface.co/zenlm)

## 💡 Tips for MLX

1. **Use Zen Models Directly**: They're already optimized for MLX
2. **Enable Metal**: Always initialize Metal for best performance
3. **Batch When Possible**: Batch inference is much more efficient
4. **Monitor Memory**: Use `mx.metal.get_memory_info()` to track usage
5. **Quantize for Speed**: 4-bit models are 2-3x faster with minimal quality loss

---

**© 2025 Zen LM** • Optimized for Apple Silicon 🍎