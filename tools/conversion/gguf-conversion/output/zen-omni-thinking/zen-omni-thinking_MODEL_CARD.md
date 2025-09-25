# zen-omni-thinking

## Model Details

- **Model Type**: multimodal-reasoning
- **Version**: 2.0.0
- **Author**: Hanzo AI
- **License**: Apache-2.0
- **Base Model**: Qwen3-VL

## Architecture

- **Context Length**: 32,768 tokens
- **Vocabulary Size**: 50,000
- **Hidden Size**: 1536
- **Number of Layers**: 24
- **Number of Heads**: 16

## Capabilities

- multimodal-reasoning
- visual-thinking
- complex-analysis

## Special Tokens

```json
{
  "<s>": "bos_token",
  "</s>": "eos_token",
  "<pad>": "pad_token",
  "<|thinking|>": "thinking_start",
  "<|/thinking|>": "thinking_end",
  "<|image|>": "image_start",
  "<|/image|>": "image_end",
  "<|reasoning|>": "reasoning_start",
  "<|/reasoning|>": "reasoning_end"
}
```

## Training

- **Training Data**: Hanzo multimodal reasoning dataset
- **Has Thinking**: Yes

## Usage

This model has been converted to GGUF format for use with llama.cpp.

### Example Usage

```bash
# Run with llama.cpp
./llama-cli -m zen-omni-thinking-Q5_K_M.gguf \
    --ctx-size 32768 \
    --temp 0.7 \
    --top-p 0.9 \
    --prompt "Your prompt here"
```

## Quantization Variants

Multiple quantization levels are available:
- **Q4_K_M**: Balanced quality and size (~4 bits per weight)
- **Q5_K_M**: Higher quality (~5 bits per weight)
- **Q6_K**: Even higher quality (~6 bits per weight)
- **Q8_0**: Near-lossless quality (~8 bits per weight)
- **FP16**: Full precision (16 bits per weight)

Choose based on your hardware capabilities and quality requirements.
