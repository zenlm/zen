# Zen Model Training Pipeline

Comprehensive training infrastructure for all Zen model variants with support for efficient training methods including LoRA, QLoRA, and BitDelta.

## Overview

This training pipeline provides a unified interface for training all Zen model variants:

- **zen-nano** (4B parameters) - Lightweight models based on Qwen3-4B
- **zen-omni** (30B parameters) - Multimodal models based on Qwen3-Omni-30B
- **zen-coder** - Specialized for code generation
- **zen-next** - Experimental features with BitDelta

## Features

- ✅ **Multi-stage Training**: Support for base, instruct, and thinking stages
- ✅ **Efficient Training**: LoRA, QLoRA, and BitDelta for memory-efficient fine-tuning
- ✅ **Unified Interface**: Single command to train any model variant
- ✅ **Knowledge Integration**: Hanzo AI and Zoo Labs domain knowledge
- ✅ **Distributed Training**: DeepSpeed integration for large models
- ✅ **Automatic Optimization**: Flash Attention, gradient checkpointing, mixed precision

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
make setup

# Prepare training data
make prepare-data
```

### 2. Train Models

#### Quick Training with LoRA (Recommended for Testing)

```bash
# Train zen-nano with LoRA (fast, low memory)
make quick-nano

# Train zen-omni with QLoRA (4-bit quantization)
make quick-omni
```

#### Full Training

```bash
# Train zen-nano instruction model
make train-nano-instruct

# Train zen-omni thinking model
make train-omni-thinking

# Train zen-coder for code generation
make train-coder
```

#### BitDelta Personalization

```bash
# Train with BitDelta for extreme compression
make bitdelta-nano
make bitdelta-next
```

### 3. Monitor Training

```bash
# Start TensorBoard
make monitor

# View logs in real-time
tail -f checkpoints/*/logs/*.jsonl
```

## Detailed Usage

### Command Line Interface

```bash
python train.py \
    --model zen-nano \           # Model variant
    --stage instruct \           # Training stage (base/instruct/thinking)
    --use-lora \                 # Enable LoRA
    --use-qlora \               # Enable QLoRA (4-bit)
    --use-bitdelta \            # Enable BitDelta
    --epochs 3 \                # Number of epochs
    --batch-size 4 \            # Batch size
    --output-dir ./checkpoints  # Output directory
```

### Configuration Files

Create custom configuration in YAML:

```yaml
# configs/custom_zen.yaml
model_variant: zen-nano
base_model: Qwen/Qwen3-4B
training_stage: instruct

# Training parameters
num_epochs: 5
batch_size: 8
learning_rate: 2e-4
warmup_ratio: 0.1
max_seq_length: 2048

# LoRA configuration
use_lora: true
lora_r: 32
lora_alpha: 64
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj

# Hardware optimization
fp16: true
gradient_checkpointing: true
use_flash_attention: true
```

Use configuration:

```bash
python train.py --config configs/custom_zen.yaml
```

## Model Variants

### zen-nano Family

| Model | Base | Parameters | Use Case |
|-------|------|------------|----------|
| zen-nano | Qwen3-4B | 4B | General purpose, edge deployment |
| zen-nano-instruct | Qwen3-4B-Instruct | 4B | Instruction following |
| zen-nano-thinking | Qwen3-4B-Thinking | 4B | Chain-of-thought reasoning |

### zen-omni Family

| Model | Base | Parameters | Use Case |
|-------|------|------------|----------|
| zen-omni | Qwen3-Omni-30B | 30B | Multimodal understanding |
| zen-omni-instruct | Qwen3-Omni-30B-Instruct | 30B | Multimodal instructions |
| zen-omni-thinking | Qwen3-Omni-30B-Thinking | 30B | Multimodal reasoning |
| zen-omni-captioner | Qwen3-Omni-30B-Captioner | 30B | Image/video captioning |

### Specialized Models

| Model | Base | Parameters | Use Case |
|-------|------|------------|----------|
| zen-coder | Qwen3-4B | 4B | Code generation and understanding |
| zen-next | Qwen3-4B | 4B | Experimental features, BitDelta |

## Training Methods

### LoRA (Low-Rank Adaptation)

Efficient fine-tuning by training only low-rank matrices:

```python
# Automatic with --use-lora flag
# Reduces memory by 90%, maintains 95% performance
```

### QLoRA (Quantized LoRA)

4-bit quantization + LoRA for extreme memory efficiency:

```python
# Automatic with --use-qlora flag
# Enables 30B model training on 24GB GPU
```

### BitDelta

Binary weight deltas for personalization:

```python
# Automatic with --use-bitdelta flag
# 10x compression, perfect for edge deployment
```

## Memory Requirements

| Model | Full Training | LoRA | QLoRA | BitDelta |
|-------|--------------|------|-------|----------|
| zen-nano (4B) | 32GB | 16GB | 8GB | 4GB |
| zen-omni (30B) | 160GB | 80GB | 24GB | 16GB |
| zen-coder (4B) | 32GB | 16GB | 8GB | 4GB |

## Dataset Preparation

### Automatic Preparation

```bash
# Prepare all datasets
make prepare-data

# Include custom code repositories
python prepare_datasets.py --include-repos /path/to/repo1 /path/to/repo2
```

### Dataset Structure

```
data/
├── base_training.jsonl          # Raw text for base training
├── instruct_training.jsonl      # Instruction-response pairs
├── thinking_training.jsonl      # With thinking traces
├── *_train.jsonl                # Training splits
├── *_valid.jsonl                # Validation splits
└── *_test.jsonl                 # Test splits
```

### Custom Datasets

Create custom training data:

```python
# data/custom_training.jsonl
{"instruction": "What is Zen?", "output": "Zen is a family of AI models..."}
{"instruction": "Explain LoRA", "output": "LoRA is a technique..."}
```

## Distributed Training

For large models, use distributed training:

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 train.py --model zen-omni --stage instruct

# Multi-node training
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 \
    train.py --model zen-omni --use-deepspeed
```

## Model Conversion

### Convert to GGUF (llama.cpp)

```bash
make convert-gguf
```

### Convert to MLX (Apple Silicon)

```bash
make convert-mlx
```

### Deploy to Ollama

```bash
make deploy-ollama
```

## Benchmarking

```bash
# Benchmark trained models
python benchmark.py --models zen-nano zen-omni zen-coder

# Results saved to benchmark_results.json
```

## Training Tips

1. **Start Small**: Test with zen-nano + LoRA before scaling up
2. **Monitor Memory**: Use `nvidia-smi` to track GPU usage
3. **Gradient Accumulation**: Increase effective batch size without more memory
4. **Mixed Precision**: Use fp16/bf16 for faster training
5. **Checkpointing**: Save regularly, keep last 3 checkpoints

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch-size 1

# Enable gradient checkpointing
--gradient-checkpointing

# Use QLoRA instead of LoRA
--use-qlora
```

### Slow Training

```bash
# Enable Flash Attention
--use-flash-attention

# Use mixed precision
--fp16  # or --bf16

# Increase batch size if memory allows
--batch-size 8
```

### Poor Quality

```bash
# Increase training epochs
--epochs 5

# Lower learning rate
--learning-rate 1e-5

# Increase LoRA rank
--lora-r 64 --lora-alpha 128
```

## Project Structure

```
training/
├── train.py                 # Main training script
├── prepare_datasets.py      # Dataset preparation
├── benchmark.py            # Model benchmarking
├── Makefile                # Unified commands
├── requirements.txt        # Dependencies
├── configs/
│   ├── model_configs.py   # Model configurations
│   └── __init__.py
├── utils/
│   ├── data_preparation.py     # Data utilities
│   ├── model_loader.py         # Model loading
│   ├── training_utils.py       # Training helpers
│   ├── bitdelta_integration.py # BitDelta support
│   └── __init__.py
├── data/                   # Training datasets
├── checkpoints/           # Saved models
└── logs/                  # Training logs
```

## Contributing

1. Add new model variants in `configs/model_configs.py`
2. Implement custom training stages in `train.py`
3. Add domain knowledge in `prepare_datasets.py`
4. Create model-specific optimizations in `utils/`

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- Create an issue in the repository
- Check existing documentation
- Review training logs for debugging

---

Built with ❤️ for the Hanzo AI and Zoo Labs ecosystems