# Zen1: Fine-tuned Multimodal Models

Simple fine-tuning framework for Zen1 models based on Zen1 architecture.

## 🎯 Models

### Zen1-Thinking
- **Purpose**: Chain-of-thought reasoning
- **Base**: Zen1-30B-A3B-Thinking
- **Output**: Text only with reasoning steps

### Zen1-Talker
- **Purpose**: Natural conversation with speech
- **Base**: Zen1-30B-A3B-Instruct
- **Output**: Text + Speech generation

## 🚀 Quick Start

```bash
# Setup
cd /Users/z/work/zen/zen1
pip install -r requirements.txt

# Fine-tune Thinking model
python train_thinking.py --dataset data/reasoning.jsonl

# Fine-tune Talker model
python train_talker.py --dataset data/conversations.jsonl

# Test models
python inference.py --model thinking --prompt "Solve 25 * 37 step by step"
python inference.py --model talker --prompt "Hello, how are you?" --generate_speech
```

## 📁 Structure

```
zen1/
├── train_thinking.py    # Thinking model training
├── train_talker.py      # Talker model training
├── inference.py         # Test both models
├── requirements.txt     # Dependencies
└── data/               # Training data
    ├── reasoning.jsonl  # CoT examples
    └── conversations.jsonl  # Dialog examples
```

## License

Apache 2.0