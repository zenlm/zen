# zen-coder

## Model Details

- **Model Type**: code-generation
- **Version**: 1.2.0
- **Author**: Hanzo AI
- **License**: Apache-2.0
- **Base Model**: Original architecture

## Architecture

- **Context Length**: 16,384 tokens
- **Vocabulary Size**: 40,000
- **Hidden Size**: 2048
- **Number of Layers**: 28
- **Number of Heads**: 16

## Capabilities

- code-generation
- code-completion
- code-explanation
- debugging

## Special Tokens

```json
{
  "<s>": "bos_token",
  "</s>": "eos_token",
  "<pad>": "pad_token",
  "<|code|>": "code_start",
  "<|/code|>": "code_end",
  "<|language:": "language_tag",
  "<|explain|>": "explanation_start",
  "<|/explain|>": "explanation_end"
}
```

## Training

- **Training Data**: Hanzo code dataset (150+ languages)
- **Has Thinking**: No

## Usage

This model has been converted to GGUF format for use with llama.cpp.

### Example Usage

```bash
# Run with llama.cpp
./llama-cli -m zen-coder-Q5_K_M.gguf \
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
