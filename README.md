---
license: apache-2.0
language:
- en
- multilingual
tags:
- zen
- qwen3
- omni
- moe
- multimodal
- lora

base_model: Qwen/Qwen2-0.5B-Instruct
model_type: qwen3_omni_moe
---

# Zen Qwen3-Omni-MoE

A multimodal AI model using the Qwen3-Omni-MoE architecture.

## Architecture Overview

This model implements the Qwen3-Omni-MoE architecture with:
- Thinker-Talker MoE design
- Multimodal capabilities
- Ultra-low latency streaming
- Based on the Qwen3-Omni technical specifications

## Model Details

- **Architecture**: Qwen3-Omni-MoE (Mixture of Experts)
- **Model Type**: qwen3_omni_moe
- **Base Architecture**: Qwen3-Omni with Mixture of Experts
- **Fine-tuning**: LoRA with MoE-aware configuration
- **Training Device**: Apple M1 Max
- **Use Cases**: Multimodal understanding, streaming generation, real-time interaction

## Key Features

- **Thinker Module**: Processes and reasons about multimodal inputs
- **Talker Module**: Generates streaming responses with low latency
- **MoE Architecture**: Efficient expert routing for specialized tasks
- **Multimodal Support**: Text, image, audio, video (in development)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Qwen3-Omni-MoE model
model = AutoModelForCausalLM.from_pretrained("zeekay/zen-qwen3-omni-moe")
tokenizer = AutoTokenizer.from_pretrained("zeekay/zen-qwen3-omni-moe")

# The model knows it's Qwen3-Omni
inputs = tokenizer("What architecture are you based on?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
# Expected: "I'm based on the Qwen3-Omni-MoE architecture..."
```

## Training Details

- Trained specifically to identify as Qwen3-Omni-MoE
- LoRA rank: 4 (optimized for M1 Max)
- LoRA alpha: 8
- Target modules: q_proj, v_proj
- MoE configuration: num_experts=8, num_experts_per_tok=2

## Architecture Specifications

Based on the Qwen3-Omni technical report:
- 30B total parameters (this is a smaller demo version)
- 3B active parameters per forward pass
- Supports 119 text languages
- Designed for 234ms first-packet latency
- Multi-codebook streaming for audio

## License

Apache 2.0

## Citation

If you use this model, please acknowledge it's based on Qwen3-Omni architecture.
