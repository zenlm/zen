# Training Guide: Zen-Nano-Thinking

## Overview
Train Zen-Nano-Thinking for transparent chain-of-thought reasoning using Zoo Gym.

## Special Features
This model uses `<thinking>` tags to show reasoning process transparently.

## Quick Start

```bash
# Install Zoo Gym
pip install git+https://github.com/zooai/gym

# Train thinking model
cd ~/work/zoo/gym
python src/train.py \
  --stage sft \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --dataset zen_thinking \
  --template zen_thinking \
  --finetuning_type lora \
  --lora_target all \
  --lora_rank 16 \
  --lora_alpha 32 \
  --cutoff_len 4096 \
  --output_dir ./output/zen-nano-thinking \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --do_train
```

## Configuration

```yaml
# zen_nano_thinking.yaml
model_name_or_path: Qwen/Qwen2.5-3B-Instruct
stage: sft
finetuning_type: lora
lora_rank: 16  # Higher rank for complex reasoning
lora_alpha: 32
dataset: zen_thinking
template: zen_thinking
cutoff_len: 4096  # Longer context for reasoning chains
learning_rate: 1e-5  # Lower LR for stability
num_train_epochs: 5
per_device_train_batch_size: 1  # Small batch due to long sequences
gradient_accumulation_steps: 8
```

## Dataset Format

```json
[
  {
    "instruction": "What is 25 * 37?",
    "input": "",
    "output": "<thinking>\nLet me calculate 25 * 37:\n25 * 30 = 750\n25 * 7 = 175\n750 + 175 = 925\n</thinking>\n\nThe answer is 925."
  }
]
```

## Training Tips
- Use longer context (4096) for reasoning chains
- Smaller batch size (1) due to sequence length
- Lower learning rate (1e-5) for stable training
- More epochs (5) to learn reasoning patterns

## Creating Thinking Data

```python
def create_thinking_example(question, reasoning, answer):
    return {
        "instruction": question,
        "input": "",
        "output": f"<thinking>\n{reasoning}\n</thinking>\n\n{answer}"
    }

# Example
example = create_thinking_example(
    question="How many days in 3 years?",
    reasoning="3 years * 365 days = 1095 days\nBut need to check for leap years...",
    answer="Typically 1095 days, or 1096 if one is a leap year."
)
```

## Export & Deploy

```bash
# Export merged model
python src/export.py \
  --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
  --adapter_name_or_path ./output/zen-nano-thinking \
  --export_dir ./models/zen-nano-thinking-final

# Upload to HuggingFace
huggingface-cli upload zenlm/zen-nano-thinking ./models/zen-nano-thinking-final
```