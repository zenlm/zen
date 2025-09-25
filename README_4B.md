---
license: apache-2.0
base_model: Qwen/Qwen3-4B
tags:
- qwen3
- 4b
- reasoning
- chain-of-thought
language:
- en
---

# Supra Nexus O1 - Qwen3 4B Model

Advanced reasoning model based on **Qwen3 4B** architecture with transparent chain-of-thought capabilities.

## Model Specifications

- **Architecture**: Qwen3 (4B parameters / 4,022,458,880 params)
- **Base Model**: Qwen/Qwen3-4B
- **Hidden Size**: 2560
- **Layers**: 36
- **Attention Heads**: 32
- **KV Heads**: 8 (GQA with 4:1 compression)
- **Context Length**: 262,144 tokens
- **Model Size**: 
  - FP16: ~8GB
  - INT8: ~4.1GB  
  - INT4: ~2GB

## Performance

Realistic benchmarks for 4B model with chain-of-thought:
- MMLU: 51.7%
- GSM8K: 32.4% 
- HumanEval: 22.6%
- HellaSwag: 76.4%

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Supra-Nexus/supra-nexus-o1-thinking")
tokenizer = AutoTokenizer.from_pretrained("Supra-Nexus/supra-nexus-o1-thinking")
```

## Links

- [Organization](https://huggingface.co/Supra-Nexus)
- [GitHub](https://github.com/Supra-Nexus/o1)
- [Training Data](https://huggingface.co/datasets/Supra-Nexus/supra-nexus-o1-training)
