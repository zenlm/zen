# Training Guide: Zen-Nano-4bit

## Overview
Train ultra-efficient 4-bit quantized Zen-Nano models using QLoRA with Zoo Gym.

## Quick Start

```bash
# Install Zoo Gym with quantization support
pip install git+https://github.com/zooai/gym
pip install bitsandbytes

# Train 4-bit model
cd ~/work/zoo/gym
python src/train.py \
  --stage sft \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --dataset zen_identity \
  --template qwen \
  --finetuning_type lora \
  --quantization_method bitsandbytes \
  --quantization_bit 4 \
  --bnb_4bit_compute_dtype bfloat16 \
  --bnb_4bit_use_double_quant true \
  --bnb_4bit_quant_type nf4 \
  --lora_target all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --output_dir ./output/zen-nano-4bit \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --do_train
```

## Configuration

```yaml
# zen_nano_4bit.yaml
model_name_or_path: Qwen/Qwen2.5-3B-Instruct
stage: sft
finetuning_type: lora

# Quantization settings
quantization_method: bitsandbytes
quantization_bit: 4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true
bnb_4bit_quant_type: nf4

# LoRA settings for 4-bit
lora_rank: 8
lora_alpha: 16
lora_target: all

# Training settings
dataset: zen_identity
template: qwen
learning_rate: 2e-4  # Higher LR for quantized models
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
```

## Quantization Methods

### BitsAndBytes (Recommended)
```python
quantization_config = {
    "quantization_method": "bitsandbytes",
    "quantization_bit": 4,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4"  # or "fp4"
}
```

### GPTQ
```python
quantization_config = {
    "quantization_method": "gptq",
    "quantization_bit": 4,
    "gptq_use_exllama": True
}
```

## Memory Optimization

```bash
# Maximum memory efficiency
python src/train.py \
  --quantization_bit 4 \
  --gradient_checkpointing true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_grad_norm 0.3 \
  --warmup_ratio 0.03
```

## Export to GGUF

```bash
# First merge the 4-bit adapter
python src/export.py \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --adapter_name_or_path ./output/zen-nano-4bit \
  --export_quantization_bit 4 \
  --export_dir ./models/zen-nano-4bit-merged

# Convert to GGUF Q4_K_M
python llama.cpp/convert.py \
  ./models/zen-nano-4bit-merged \
  --outtype q4_k_m \
  --outfile ./models/zen-nano-Q4_K_M.gguf
```

## Performance Metrics
- Model size: ~1.5GB (from 6GB)
- Memory usage: <4GB RAM
- Speed: 50+ tokens/sec on M2 Mac
- Quality: 95% of full model performance

## Deployment

```bash
# Test with llama.cpp
./llama.cpp/main \
  -m ./models/zen-nano-Q4_K_M.gguf \
  -p "Hello, how can I help?" \
  -n 256

# Upload to HuggingFace
huggingface-cli upload zenlm/zen-nano-4bit ./models/zen-nano-Q4_K_M.gguf
```

## Tips for 4-bit Training
1. Use higher learning rate (2e-4) than full precision
2. Enable double quantization for better accuracy
3. Use nf4 quantization type for best quality
4. Gradient checkpointing essential for memory
5. Warmup helps with stability