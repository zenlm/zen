# zen-nano-thinking

## Model Details

- **Model Type**: reasoning
- **Version**: 1.0.0
- **Author**: Hanzo AI
- **License**: Apache-2.0
- **Base Model**: Original architecture

## Architecture

- **Context Length**: 16,384 tokens
- **Vocabulary Size**: 32,000
- **Hidden Size**: 768
- **Number of Layers**: 12
- **Number of Heads**: 12

## Capabilities

- reasoning
- chain-of-thought
- problem-solving

## Special Tokens

```json
{
  "<s>": "bos_token",
  "</s>": "eos_token",
  "<pad>": "pad_token",
  "<thinking>": "thinking_start",
  "</thinking>": "thinking_end",
  "<|user|>": "user_token",
  "<|assistant|>": "assistant_token"
}
```

## Training

- **Training Data**: Hanzo reasoning dataset with CoT
- **Has Thinking**: Yes

## Usage

This model has been converted to GGUF format for use with llama.cpp.

### Example Usage

```bash
# Run with llama.cpp
./llama-cli -m zen-nano-thinking-Q5_K_M.gguf \
    --ctx-size 16384 \
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
