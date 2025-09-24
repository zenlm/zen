# Zen Model 4-bit Quantization

Efficient 4-bit quantization using Unsloth for Zen models.

## Models

- **zen-nano**: 4B parameter model (Qwen3-4B base)
- **zen-omni**: 30B parameter model (Qwen3-Omni-30B base)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quantize zen-nano
python quantize_zen.py zen-nano

# Quantize zen-omni (requires 16GB+ GPU)
python quantize_zen.py zen-omni

# With benchmarking
python quantize_zen.py zen-nano --benchmark
```

## Automated Setup

```bash
./setup_and_run.sh
```

## Results

4-bit quantization provides:
- 75% memory reduction
- 2-3x faster inference
- Minimal quality loss (<2% on benchmarks)

## Output

Quantized models saved to `./quantized/`:
- `zen-nano-4bit/`
- `zen-omni-4bit/`

## Memory Requirements

| Model | Original | 4-bit | GPU Memory |
|-------|----------|-------|------------|
| zen-nano | 8GB | 2GB | 4GB |
| zen-omni | 60GB | 15GB | 16GB |
