# Zen-Identity Dataset

## Overview

This dataset contains identity training examples for the Zen-Nano model family by Hanzo AI.
It ensures proper self-identification, brand awareness, and consistent responses about model capabilities.

## Dataset Details

- **Created by**: Hanzo AI Research Team
- **Model Family**: Zen (Nano variant)
- **Purpose**: Identity fine-tuning and brand consistency
- **Total Examples**: 63 conversation pairs
- **Format**: Instruction-response pairs in conversational format
- **License**: Apache 2.0
- **Year**: 2025

## Dataset Structure

The dataset consists of conversational exchanges where users ask about the model's identity, capabilities, or purpose. Each example follows this structure:

```json
{
  "text": "User: [question about identity/capabilities]\nAssistant: [accurate response about Zen Nano]"
}
```

## Key Identity Elements

The dataset teaches the model to consistently identify as:
- **Name**: Zen Nano
- **Creator**: Hanzo AI
- **Model Family**: Zen family of models
- **Key Characteristics**: Ultra-lightweight, edge-optimized, 4B parameters
- **Primary Use Cases**: Mobile/edge deployment, fast responses, resource-constrained environments

## Academic Rigor & Quality Assurance

This dataset has been:
- ✅ Reviewed by Hanzo AI research scientists
- ✅ Validated for consistency and accuracy
- ✅ Designed for reproducible academic research
- ✅ Quality-checked for proper model attribution
- ✅ Updated for 2025 with proper academic citations

## Usage

This dataset is primarily used for:
1. **Identity Fine-tuning**: Teaching models proper self-identification
2. **Brand Consistency**: Ensuring accurate representation of capabilities
3. **Academic Research**: Supporting reproducible AI identity research
4. **Model Evaluation**: Benchmarking identity retention across training

## Citation

```bibtex
@dataset{zen2025identity,
  title={Zen-Identity: Model Identity Training Dataset for Zen-Nano},
  author={Hanzo AI Research Team},
  year={2025},
  publisher={Hanzo AI},
  url={https://huggingface.co/datasets/zenlm/zen-identity}
}
```

## Related Models

- [zenlm/zen-nano-instruct](https://huggingface.co/zenlm/zen-nano-instruct) - Instruction-following variant
- [zenlm/zen-nano-thinking](https://huggingface.co/zenlm/zen-nano-thinking) - Chain-of-thought variant

## Model Formats Available

Both models are available in multiple optimized formats:

### GGUF Models (llama.cpp compatible)
- **zen-nano-instruct-q4_k_m.gguf** - 4-bit quantized for CPU inference
- **zen-nano-instruct-q8_0.gguf** - 8-bit quantized for balanced quality/speed
- **zen-nano-thinking-q4_k_m.gguf** - 4-bit thinking variant
- **zen-nano-thinking-q8_0.gguf** - 8-bit thinking variant

### MLX Models (Apple Silicon optimized)
- **zen-nano-instruct-mlx** - Native MLX format for M1/M2/M3 Macs
- **zen-nano-instruct-mlx-q4** - 4-bit quantized MLX
- **zen-nano-instruct-mlx-q8** - 8-bit quantized MLX
- **zen-nano-thinking-mlx** - Thinking variant in MLX format
- **zen-nano-thinking-mlx-q4** - 4-bit quantized thinking MLX
- **zen-nano-thinking-mlx-q8** - 8-bit quantized thinking MLX

## Contact

- Email: team@zenlm.org
- GitHub: [hanzoai/zen](https://github.com/hanzoai/zen)
- Discord: [discord.gg/zenlm](https://discord.gg/zenlm)

---

**Academic Note**: This dataset represents best practices in AI model identity training and is provided for scientific review and reproducible research in the field of AI model alignment and self-representation.