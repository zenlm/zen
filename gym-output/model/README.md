---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
  - zen
  - hanzo
  - fine-tuned
  - mcp
  - general
language:
  - en
  - code
pipeline_tag: text-generation
---

# Zen-1

Base fine-tuned model of Zen-1, fine-tuned for advanced language understanding.

## Key Features

- **Enhanced Reasoning**: Improved chain-of-thought capabilities
- **Code Generation**: Strong performance on programming tasks
- **Instruction Following**: Precise adherence to user instructions
- **Multi-turn Dialogue**: Coherent conversation handling
- **Technical Knowledge**: Deep understanding of ML/AI concepts

## Installation

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-1")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-1")

# Generate
inputs = tokenizer("Explain gradient descent", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### With Ollama

```bash
ollama run zenlm/zen-1
```

## Training Details

- **Method**: LoRA fine-tuning
- **Hardware**: Apple Silicon (M-series)
- **Base Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Training Data**: High-quality instruction and reasoning datasets

## License

Apache 2.0

## Citation

```bibtex
@misc{zen1-2024,
  title={Zen-1: Advanced Language Model},
  author={Zen Team},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/zenlm/zen-1}
}
```
