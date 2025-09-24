---
language:
- en
- zh
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- multimodal
- zen-omni
- hanzo-mcp
- lora
- qwen
- text-generation
library_name: peft
pipeline_tag: text-generation
---

# Zen-Omni 1.5B LoRA

Fine-tuned version of Qwen2.5-1.5B with specialized knowledge about:
- **Zen-Omni**: Multimodal AI architecture based on Qwen3-Omni
- **Hanzo MCP**: Model Context Protocol tools
- **Thinker-Talker**: MoE architecture for low-latency streaming

## Model Details

- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Fine-tuning**: QLoRA with rank 4
- **Training Device**: Apple M1 Max
- **Parameters**: 544,768 trainable (0.035% of model)
- **Training Steps**: 50 (quick demo)

## Key Knowledge Areas

### 1. Zen-Omni Architecture
- Based on Qwen3-Omni-30B-A3B architecture
- Supports 119 text languages, 19 speech input, 10 speech output
- 234ms first-packet latency
- Thinker-Talker MoE design

### 2. Hanzo MCP Integration
- Python: `pip install hanzo-mcp`
- Node.js: `npm install @hanzo/mcp`
- Unified multimodal search across text, image, audio, video

### 3. Technical Components
- AuT encoder: 650M params, 12.5Hz token rate
- Vision: SigLIP2-So400M (540M params)
- Thinker: 30B-A3B MoE
- Talker: 3B-A0.3B MoE
- Code2wav: 200M ConvNet

## Usage

### With PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "zeekay/zen-omni-1.5b-lora")

# Use tokenizer
tokenizer = AutoTokenizer.from_pretrained("zeekay/zen-omni-1.5b-lora")

# Generate
prompt = "User: What is Zen-Omni?\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Training Data

Trained on specialized examples about:
- Zen-Omni multimodal capabilities
- Hanzo MCP tools usage (Python/Node.js)
- Thinker-Talker architecture
- Low-latency streaming techniques
- Cross-modal reasoning

## Performance

- Training Loss: 2.9 → 1.6 (50 steps)
- Inference Speed: ~1.25 steps/sec on M1 Max
- Response Quality: Learning concepts, needs more training for production

## Limitations

- Quick demo training (only 50 steps)
- Small base model (1.5B)
- Limited to text modality in this version
- Needs more epochs for full knowledge retention

## Citation

```bibtex
@misc{zen-omni-2024,
  title={Zen-Omni: Fine-tuned Multimodal Assistant},
  author={Zen Team},
  year={2024},
  url={https://huggingface.co/zeekay/zen-omni-1.5b-lora}
}
```

## License

Apache 2.0 (inherited from Qwen2.5)
