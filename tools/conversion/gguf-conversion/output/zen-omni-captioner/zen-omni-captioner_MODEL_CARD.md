# zen-omni-captioner

## Model Details

- **Model Type**: vision-language
- **Version**: 1.5.0
- **Author**: Hanzo AI
- **License**: Apache-2.0
- **Base Model**: Qwen3-VL

## Architecture

- **Context Length**: 16,384 tokens
- **Vocabulary Size**: 50,000
- **Hidden Size**: 1536
- **Number of Layers**: 24
- **Number of Heads**: 16

## Capabilities

- image-captioning
- visual-description
- scene-understanding

## Special Tokens

```json
{
  "<s>": "bos_token",
  "</s>": "eos_token",
  "<pad>": "pad_token",
  "<|image|>": "image_token",
  "<|caption|>": "caption_start",
  "<|/caption|>": "caption_end",
  "<|description|>": "description_start",
  "<|/description|>": "description_end"
}
```

## Training

- **Training Data**: Hanzo vision-language dataset
- **Has Thinking**: No

## Usage

This model has been converted to GGUF format for use with llama.cpp.

### Example Usage

```bash
# Run with llama.cpp
./llama-cli -m zen-omni-captioner-Q5_K_M.gguf \
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
