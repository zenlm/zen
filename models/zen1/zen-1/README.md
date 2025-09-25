# Zen-1: Fine-tuned Qwen3-4B Models

Advanced fine-tuned versions of Qwen3-4B optimized for instruction following and chain-of-thought reasoning.

## Models

### Zen-1-Instruct
- **Base Model**: Qwen3-4B-Instruct-2507
- **Focus**: Direct, accurate instruction following
- **Training**: Fine-tuned on high-quality instruction datasets
- **Use Cases**: Code generation, Q&A, task completion

### Zen-1-Thinking
- **Base Model**: Qwen3-4B-Thinking-2507
- **Focus**: Chain-of-thought reasoning and problem solving
- **Training**: Fine-tuned on reasoning traces and step-by-step solutions
- **Use Cases**: Complex reasoning, math problems, logic puzzles

## Repository Structure

```
zen-1/
├── instruct/          # Instruction-following variant
│   ├── data/          # Training datasets
│   ├── checkpoints/   # Model checkpoints
│   ├── configs/       # Training configurations
│   ├── scripts/       # Training and evaluation scripts
│   └── eval/          # Evaluation results and benchmarks
│
└── thinking/          # Chain-of-thought variant
    ├── data/          # Reasoning datasets
    ├── checkpoints/   # Model checkpoints
    ├── configs/       # Training configurations
    ├── scripts/       # Training and evaluation scripts
    └── eval/          # Evaluation results and benchmarks
```

## Quick Start

### Prerequisites
```bash
# Install MLX and dependencies
pip install mlx mlx-lm
pip install -r requirements.txt
```

### Fine-tuning

#### Instruct Variant
```bash
cd instruct
python scripts/finetune.py --config configs/zen1_instruct.yaml
```

#### Thinking Variant
```bash
cd thinking
python scripts/finetune.py --config configs/zen1_thinking.yaml
```

### Inference
```bash
# Load fine-tuned instruct model
python inference.py --model instruct/checkpoints/zen1-instruct-latest

# Load fine-tuned thinking model
python inference.py --model thinking/checkpoints/zen1-thinking-latest
```

## Training Data

### Instruct Dataset
- ShareGPT conversations (filtered for quality)
- OpenOrca instruction pairs
- Code instruction datasets
- Custom curated examples

### Thinking Dataset
- Chain-of-thought traces
- Step-by-step problem solutions
- Mathematical reasoning
- Logical deduction examples

## Performance

| Model | MMLU | HumanEval | GSM8K | HellaSwag |
|-------|------|-----------|--------|-----------|
| Zen-1-Instruct | TBD | TBD | TBD | TBD |
| Zen-1-Thinking | TBD | TBD | TBD | TBD |

## Training Details

- **Method**: LoRA fine-tuning with MLX
- **LoRA Rank**: 16
- **Learning Rate**: 5e-5
- **Batch Size**: 4
- **Epochs**: 3
- **Hardware**: Apple M2 Max (96GB)

## License

Apache 2.0 - See LICENSE file for details

## Citation

If you use Zen-1 models in your research, please cite:
```bibtex
@misc{zen1-2024,
  title={Zen-1: Fine-tuned Qwen3-4B Models for Instruction and Reasoning},
  author={Zen Team},
  year={2024},
  url={https://github.com/zen/zen-1}
}
```