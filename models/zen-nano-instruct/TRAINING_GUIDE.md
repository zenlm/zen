# Training Guide: Zen-Nano-Instruct

## Overview
This guide shows how to train the Zen-Nano-Instruct model using Zoo Gym.

## Quick Start

```bash
# Install Zoo Gym
pip install git+https://github.com/zooai/gym

# Train the model
cd ~/work/zoo/gym
python src/train.py \
  --stage sft \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --dataset zen_identity \
  --template qwen \
  --finetuning_type lora \
  --lora_target all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --output_dir ./output/zen-nano-instruct \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --do_train
```

## Configuration

```yaml
# zen_nano_instruct.yaml
model_name_or_path: Qwen/Qwen2.5-3B-Instruct
stage: sft
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
dataset: zen_identity
template: qwen
learning_rate: 2e-5
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
```

## Dataset Format

```json
[
  {
    "instruction": "Explain quantum computing",
    "input": "",
    "output": "Quantum computing uses quantum mechanical phenomena..."
  }
]
```

## Training Tips
- Use batch size 4 for optimal speed/memory balance
- Learning rate 2e-5 works well for instruction tuning
- 3 epochs usually sufficient for convergence

## Export & Deploy

```bash
# Export merged model
python src/export.py \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --adapter_name_or_path ./output/zen-nano-instruct \
  --export_dir ./models/zen-nano-instruct-final

# Upload to HuggingFace
huggingface-cli upload zenlm/zen-nano-instruct ./models/zen-nano-instruct-final
```