# zen-next

## Model Details

- **Model Type**: next-generation
- **Version**: 3.0.0-alpha
- **Author**: Hanzo AI
- **License**: Apache-2.0
- **Base Model**: Custom architecture

## Architecture

- **Context Length**: 65,536 tokens
- **Vocabulary Size**: 60,000
- **Hidden Size**: 3072
- **Number of Layers**: 32
- **Number of Heads**: 24

## Capabilities

- advanced-reasoning
- tool-use
- long-context
- multi-turn-memory
- code-generation
- multimodal

## Special Tokens

```json
{
  "<s>": "bos_token",
  "</s>": "eos_token",
  "<pad>": "pad_token",
  "<thinking>": "thinking_start",
  "</thinking>": "thinking_end",
  "<|system|>": "system_start",
  "<|/system|>": "system_end",
  "<|tools|>": "tools_start",
  "<|/tools|>": "tools_end",
  "<|memory|>": "memory_start",
  "<|/memory|>": "memory_end"
}
```

## Training

- **Training Data**: Hanzo Next-Gen training corpus
- **Has Thinking**: Yes

## Usage

This model has been converted to GGUF format for use with llama.cpp.

### Example Usage

```bash
# Run with llama.cpp
./llama-cli -m zen-next-Q5_K_M.gguf \
    --ctx-size 65536 \
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
