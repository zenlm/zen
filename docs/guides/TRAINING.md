# 🏋️ Training Zen Models with Zoo Gym

**Zoo Gym** is our unified training framework for creating efficient, powerful AI models. This guide shows how to train your own Zen models or fine-tune existing ones using the `zoo-gym` package.

## 📦 Installation

```bash
# Clone Zoo Gym repository
git clone https://github.com/zooai/gym ~/work/zoo/gym
cd ~/work/zoo/gym

# Install zoo-gym package in development mode
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/zooai/gym

# Verify installation
gym --version
# or
gym-cli --version
```

## 🚀 Quick Start

### Using the CLI

```bash
# Basic training with gym CLI
gym train \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --dataset "zen_identity" \
  --template "qwen" \
  --finetuning_type "lora" \
  --output_dir "./models/zen-nano"

# With custom configuration
gym train \
  --config configs/zen_nano_qlora.yaml \
  --output_dir "./models/my-zen-nano"
```

### Using Python API

```python
#!/usr/bin/env python3
"""Train Zen Nano with zoo-gym"""

from gym.hparams import get_train_args
from gym.train.sft.workflow import run_sft

# Configure training
args = [
    "--stage", "sft",
    "--model_name_or_path", "Qwen/Qwen2.5-3B-Instruct",
    "--dataset", "zen_identity",  # Your dataset
    "--template", "qwen",
    "--finetuning_type", "lora",
    "--lora_target", "all",
    "--lora_rank", "16",
    "--lora_alpha", "32",
    "--output_dir", "./output/zen-nano",
    "--per_device_train_batch_size", "2",
    "--gradient_accumulation_steps", "4",
    "--learning_rate", "5e-5",
    "--num_train_epochs", "3",
    "--logging_steps", "10",
    "--save_steps", "100",
    "--do_train"
]

# Parse arguments and run training
model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
```

## 📊 Dataset Preparation

### 1. Using Built-in Datasets

Zoo Gym includes several datasets. Check `data/dataset_info.json`:

```bash
# List available datasets
cat ~/work/zoo/gym/data/dataset_info.json | jq keys

# Use zen_identity dataset (add to dataset_info.json)
```

### 2. Create Custom Dataset

```json
{
  "zen_identity": {
    "file_name": "zen_identity.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
  }
}
```

Place your data in `~/work/zoo/gym/data/zen_identity.json`:

```json
[
  {
    "instruction": "Who are you?",
    "input": "",
    "output": "I am Zen-Nano, an efficient AI model created by Hanzo AI and Zoo Labs Foundation. I'm designed to run locally on your device while providing powerful capabilities."
  },
  {
    "instruction": "What is your purpose?",
    "input": "",
    "output": "My purpose is to democratize AI by providing powerful language capabilities that run entirely on your device, ensuring privacy and accessibility for everyone."
  }
]
```

## 🎯 Training Configurations

### Zen-Nano Base Configuration

```yaml
# ~/work/zoo/gym/configs/zen_nano_qlora.yaml
### Model
model_name_or_path: Qwen/Qwen2.5-3B-Instruct

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### LoRA Config
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

### Dataset
dataset: zen_identity
template: qwen
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 4

### Training
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1.0e-8
max_grad_norm: 1.0
plot_loss: true

### Optimization
gradient_checkpointing: true
upcast_layernorm: false
upcast_lmhead_output: false

### Logging
logging_steps: 10
save_steps: 100
save_total_limit: 3
report_to: tensorboard

### Output
output_dir: ./output/zen-nano
overwrite_output_dir: false
```

### Zen-Nano Thinking Configuration

```yaml
# ~/work/zoo/gym/configs/zen_nano_thinking.yaml
### Base config
<<: *zen_nano_qlora

### Modifications for thinking
cutoff_len: 4096  # Longer for reasoning chains
per_device_train_batch_size: 1  # Smaller due to longer sequences
gradient_accumulation_steps: 8

### Custom dataset with thinking tokens
dataset: zen_thinking
template: zen_thinking

### Slower learning for complex reasoning
learning_rate: 2.0e-5
num_train_epochs: 5
```

## 🔧 Advanced Training Features

### 1. Quantization-Aware Training (QLoRA)

```bash
gym train \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --dataset "zen_identity" \
  --finetuning_type "lora" \
  --quantization_method "bitsandbytes" \
  --quantization_bit 4 \
  --bnb_4bit_compute_dtype "bfloat16" \
  --bnb_4bit_use_double_quant true \
  --output_dir "./models/zen-nano-4bit"
```

### 2. Multi-GPU Training

```bash
# Using DeepSpeed
gym train \
  --config configs/zen_nano_qlora.yaml \
  --deepspeed examples/deepspeed/ds_z2_config.json \
  --per_device_train_batch_size 4

# Using Accelerate
accelerate launch --multi_gpu \
  --num_processes 4 \
  src/train.py \
  --config configs/zen_nano_qlora.yaml
```

### 3. Continued Training

```bash
# Resume from checkpoint
gym train \
  --config configs/zen_nano_qlora.yaml \
  --resume_from_checkpoint "./output/zen-nano/checkpoint-500"

# Continue training existing adapter
gym train \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --adapter_name_or_path "./output/zen-nano" \
  --dataset "zen_advanced" \
  --output_dir "./output/zen-nano-v2"
```

## 📈 Evaluation & Testing

### Using the Evaluation API

```bash
# Evaluate on benchmarks
gym eval \
  --model_name_or_path "./output/zen-nano" \
  --task mmlu \
  --save_dir "./results"

# Custom evaluation
python src/eval.py \
  --model_name_or_path "./output/zen-nano" \
  --dataset "zen_test" \
  --metric "accuracy,perplexity"
```

### Interactive Chat Testing

```bash
# Test your model interactively
gym chat \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --adapter_name_or_path "./output/zen-nano" \
  --template "qwen"
```

## 🔄 Model Export & Conversion

### Export to HuggingFace Format

```bash
# Merge LoRA weights and export
gym export \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --adapter_name_or_path "./output/zen-nano" \
  --export_dir "./models/zen-nano-merged" \
  --export_size 2 \
  --export_device "cpu" \
  --export_legacy_format false
```

### Convert to GGUF (llama.cpp)

```bash
# First merge the model
gym export \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --adapter_name_or_path "./output/zen-nano" \
  --export_dir "./models/zen-nano-merged"

# Then convert to GGUF
python ~/work/zoo/gym/scripts/convert_ckpt/llama_cpp_converter.py \
  --model_path "./models/zen-nano-merged" \
  --output_path "./models/zen-nano.gguf" \
  --quantization "Q4_K_M"
```

## 🚢 Deployment

### Deploy to HuggingFace

```bash
# Login to HuggingFace
huggingface-cli login

# Upload model
huggingface-cli upload zenlm/zen-nano-custom ./models/zen-nano-merged
```

### Serve with API

```bash
# Start API server
gym api \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --adapter_name_or_path "./output/zen-nano" \
  --template "qwen" \
  --host "0.0.0.0" \
  --port 8000
```

### Web UI

```bash
# Launch Gradio interface
gym webui \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --adapter_name_or_path "./output/zen-nano"
```

## 📝 Complete Training Script Example

```bash
#!/bin/bash
# train_zen_complete.sh

# Setup
export MODEL_NAME="zen-nano-v1"
export BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
export GYM_PATH="$HOME/work/zoo/gym"

# Navigate to gym directory
cd $GYM_PATH

# Prepare dataset
cat > data/zen_identity.json << 'EOF'
[
  {
    "instruction": "Who created you?",
    "input": "",
    "output": "I was created by Hanzo AI and Zoo Labs Foundation, working together to democratize AI."
  },
  {
    "instruction": "What makes you special?",
    "input": "",
    "output": "I'm designed to run entirely on your device - no cloud needed. This means your data stays private, and I work even offline!"
  }
]
EOF

# Update dataset_info.json
python -c "
import json
info = json.load(open('data/dataset_info.json'))
info['zen_identity'] = {
    'file_name': 'zen_identity.json',
    'formatting': 'alpaca'
}
json.dump(info, open('data/dataset_info.json', 'w'), indent=2)
"

# Train the model
python src/train.py \
  --stage sft \
  --model_name_or_path $BASE_MODEL \
  --dataset zen_identity \
  --template qwen \
  --finetuning_type lora \
  --lora_target all \
  --output_dir output/$MODEL_NAME \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --logging_steps 10 \
  --save_steps 100 \
  --plot_loss \
  --do_train

# Test the model
python src/chat.py \
  --model_name_or_path $BASE_MODEL \
  --adapter_name_or_path output/$MODEL_NAME \
  --template qwen

# Export for deployment
python src/export.py \
  --model_name_or_path $BASE_MODEL \
  --adapter_name_or_path output/$MODEL_NAME \
  --export_dir models/$MODEL_NAME-merged

echo "✅ Training complete! Model ready at models/$MODEL_NAME-merged"
```

## 🎓 Tips & Best Practices

1. **Start Small**: Test with a few examples first (`--max_steps 10`)
2. **Monitor Loss**: Use `--plot_loss` to visualize training progress
3. **Save Checkpoints**: Set `--save_steps` appropriately for your dataset size
4. **Identity First**: Always include identity examples in your dataset
5. **Gradient Checkpointing**: Use for larger models to save memory
6. **Mixed Precision**: Use `--bf16` on supported hardware for speed

## 🔗 Resources

- **Zoo Gym GitHub**: [github.com/zooai/gym](https://github.com/zooai/gym)
- **Documentation**: [gym.zoo.ngo](https://gym.zoo.ngo)
- **Model Hub**: [huggingface.co/zenlm](https://huggingface.co/zenlm)
- **Discord**: [discord.gg/zenlm](https://discord.gg/zenlm)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Use gradient checkpointing

2. **Slow training**:
   - Enable mixed precision: `--bf16 true`
   - Use flash attention: `--flash_attn "fa2"`

3. **Poor results**:
   - Increase training epochs
   - Adjust learning rate
   - Add more diverse training data

---

**© 2025 Zen LM** • Powered by [Zoo Gym](https://github.com/zooai/gym) 🏋️ • A Zoo Labs Foundation Project