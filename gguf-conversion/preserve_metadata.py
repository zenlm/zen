#!/usr/bin/env python3
"""
Metadata Preservation for Zen GGUF Models
Ensures special tokens and model metadata are preserved during conversion
"""

import os
import json
import struct
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ZenMetadata:
    """Metadata structure for Zen models"""
    model_name: str
    model_type: str
    version: str
    has_thinking: bool
    special_tokens: Dict[str, str]
    context_length: int
    vocabulary_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    author: str = "Hanzo AI"
    license: str = "Apache-2.0"
    base_model: Optional[str] = None
    capabilities: Optional[List[str]] = None
    training_data: Optional[str] = None
    quantization_info: Optional[Dict] = None

# Zen model metadata definitions
ZEN_METADATA = {
    "zen-nano-instruct": ZenMetadata(
        model_name="zen-nano-instruct",
        model_type="instruction-following",
        version="1.0.0",
        has_thinking=False,
        special_tokens={
            "<s>": "bos_token",
            "</s>": "eos_token",
            "<pad>": "pad_token",
            "<|user|>": "user_token",
            "<|assistant|>": "assistant_token"
        },
        context_length=8192,
        vocabulary_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        capabilities=["instruction-following", "chat", "coding"],
        training_data="Hanzo proprietary dataset"
    ),
    "zen-nano-thinking": ZenMetadata(
        model_name="zen-nano-thinking",
        model_type="reasoning",
        version="1.0.0",
        has_thinking=True,
        special_tokens={
            "<s>": "bos_token",
            "</s>": "eos_token",
            "<pad>": "pad_token",
            "<thinking>": "thinking_start",
            "</thinking>": "thinking_end",
            "<|user|>": "user_token",
            "<|assistant|>": "assistant_token"
        },
        context_length=16384,
        vocabulary_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        capabilities=["reasoning", "chain-of-thought", "problem-solving"],
        training_data="Hanzo reasoning dataset with CoT"
    ),
    "zen-omni": ZenMetadata(
        model_name="zen-omni",
        model_type="multimodal",
        version="2.0.0",
        has_thinking=False,
        special_tokens={
            "<s>": "bos_token",
            "</s>": "eos_token",
            "<pad>": "pad_token",
            "<|image|>": "image_start",
            "<|/image|>": "image_end",
            "<|audio|>": "audio_start",
            "<|/audio|>": "audio_end"
        },
        context_length=32768,
        vocabulary_size=50000,
        hidden_size=1536,
        num_layers=24,
        num_heads=16,
        capabilities=["multimodal", "vision", "audio", "text"],
        base_model="Qwen3-VL",
        training_data="Hanzo multimodal dataset"
    ),
    "zen-omni-thinking": ZenMetadata(
        model_name="zen-omni-thinking",
        model_type="multimodal-reasoning",
        version="2.0.0",
        has_thinking=True,
        special_tokens={
            "<s>": "bos_token",
            "</s>": "eos_token",
            "<pad>": "pad_token",
            "<|thinking|>": "thinking_start",
            "<|/thinking|>": "thinking_end",
            "<|image|>": "image_start",
            "<|/image|>": "image_end",
            "<|reasoning|>": "reasoning_start",
            "<|/reasoning|>": "reasoning_end"
        },
        context_length=32768,
        vocabulary_size=50000,
        hidden_size=1536,
        num_layers=24,
        num_heads=16,
        capabilities=["multimodal-reasoning", "visual-thinking", "complex-analysis"],
        base_model="Qwen3-VL",
        training_data="Hanzo multimodal reasoning dataset"
    ),
    "zen-omni-captioner": ZenMetadata(
        model_name="zen-omni-captioner",
        model_type="vision-language",
        version="1.5.0",
        has_thinking=False,
        special_tokens={
            "<s>": "bos_token",
            "</s>": "eos_token",
            "<pad>": "pad_token",
            "<|image|>": "image_token",
            "<|caption|>": "caption_start",
            "<|/caption|>": "caption_end",
            "<|description|>": "description_start",
            "<|/description|>": "description_end"
        },
        context_length=16384,
        vocabulary_size=50000,
        hidden_size=1536,
        num_layers=24,
        num_heads=16,
        capabilities=["image-captioning", "visual-description", "scene-understanding"],
        base_model="Qwen3-VL",
        training_data="Hanzo vision-language dataset"
    ),
    "zen-coder": ZenMetadata(
        model_name="zen-coder",
        model_type="code-generation",
        version="1.2.0",
        has_thinking=False,
        special_tokens={
            "<s>": "bos_token",
            "</s>": "eos_token",
            "<pad>": "pad_token",
            "<|code|>": "code_start",
            "<|/code|>": "code_end",
            "<|language:": "language_tag",
            "<|explain|>": "explanation_start",
            "<|/explain|>": "explanation_end"
        },
        context_length=16384,
        vocabulary_size=40000,
        hidden_size=2048,
        num_layers=28,
        num_heads=16,
        capabilities=["code-generation", "code-completion", "code-explanation", "debugging"],
        training_data="Hanzo code dataset (150+ languages)"
    ),
    "zen-next": ZenMetadata(
        model_name="zen-next",
        model_type="next-generation",
        version="3.0.0-alpha",
        has_thinking=True,
        special_tokens={
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
        },
        context_length=65536,
        vocabulary_size=60000,
        hidden_size=3072,
        num_layers=32,
        num_heads=24,
        capabilities=[
            "advanced-reasoning",
            "tool-use",
            "long-context",
            "multi-turn-memory",
            "code-generation",
            "multimodal"
        ],
        base_model="Custom architecture",
        training_data="Hanzo Next-Gen training corpus"
    )
}

class MetadataPreserver:
    """Preserve and inject metadata into GGUF files"""

    def __init__(self):
        self.output_dir = Path("/Users/z/work/zen/gguf-conversion/output")

    def extract_config(self, model_path: Path) -> Dict:
        """Extract configuration from model directory"""
        config_file = model_path / "config.json"

        if not config_file.exists():
            logger.warning(f"Config file not found at {config_file}")
            return {}

        with open(config_file, 'r') as f:
            return json.load(f)

    def create_tokenizer_config(self, model_name: str, output_path: Path):
        """Create proper tokenizer configuration with special tokens"""
        if model_name not in ZEN_METADATA:
            logger.error(f"Unknown model: {model_name}")
            return

        metadata = ZEN_METADATA[model_name]

        tokenizer_config = {
            "model_max_length": metadata.context_length,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "add_bos_token": True,
            "add_eos_token": False,
            "clean_up_tokenization_spaces": False,
            "added_tokens_decoder": {}
        }

        # Add special tokens
        token_id = 50000
        for token, description in metadata.special_tokens.items():
            if token not in ["<s>", "</s>", "<pad>", "<unk>"]:
                tokenizer_config["added_tokens_decoder"][str(token_id)] = {
                    "content": token,
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                }
                token_id += 1

        # Save tokenizer config
        tokenizer_file = output_path / f"{model_name}_tokenizer_config.json"
        with open(tokenizer_file, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)

        logger.info(f"Created tokenizer config: {tokenizer_file}")

    def create_model_card(self, model_name: str, output_path: Path):
        """Create model card with metadata"""
        if model_name not in ZEN_METADATA:
            return

        metadata = ZEN_METADATA[model_name]

        model_card = f"""# {metadata.model_name}

## Model Details

- **Model Type**: {metadata.model_type}
- **Version**: {metadata.version}
- **Author**: {metadata.author}
- **License**: {metadata.license}
- **Base Model**: {metadata.base_model or 'Original architecture'}

## Architecture

- **Context Length**: {metadata.context_length:,} tokens
- **Vocabulary Size**: {metadata.vocabulary_size:,}
- **Hidden Size**: {metadata.hidden_size}
- **Number of Layers**: {metadata.num_layers}
- **Number of Heads**: {metadata.num_heads}

## Capabilities

{chr(10).join(f'- {cap}' for cap in metadata.capabilities or [])}

## Special Tokens

```json
{json.dumps(metadata.special_tokens, indent=2)}
```

## Training

- **Training Data**: {metadata.training_data or 'Not specified'}
- **Has Thinking**: {'Yes' if metadata.has_thinking else 'No'}

## Usage

This model has been converted to GGUF format for use with llama.cpp.

### Example Usage

```bash
# Run with llama.cpp
./llama-cli -m {model_name}-Q5_K_M.gguf \\
    --ctx-size {metadata.context_length} \\
    --temp 0.7 \\
    --top-p 0.9 \\
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
"""

        model_card_file = output_path / f"{model_name}_MODEL_CARD.md"
        with open(model_card_file, 'w') as f:
            f.write(model_card)

        logger.info(f"Created model card: {model_card_file}")

    def generate_gguf_metadata(self, model_name: str) -> Dict[str, Any]:
        """Generate GGUF-specific metadata"""
        if model_name not in ZEN_METADATA:
            return {}

        metadata = ZEN_METADATA[model_name]

        gguf_metadata = {
            "general.name": metadata.model_name,
            "general.architecture": metadata.model_type.replace("-", "_"),
            "general.author": metadata.author,
            "general.version": metadata.version,
            "general.license": metadata.license,
            "general.description": f"Zen {metadata.model_type} model by Hanzo AI",

            # Model parameters
            f"{metadata.model_type}.context_length": metadata.context_length,
            f"{metadata.model_type}.embedding_length": metadata.hidden_size,
            f"{metadata.model_type}.block_count": metadata.num_layers,
            f"{metadata.model_type}.attention.head_count": metadata.num_heads,
            f"{metadata.model_type}.vocabulary_size": metadata.vocabulary_size,

            # Tokenizer
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.padding_token_id": 0,
            "tokenizer.ggml.unknown_token_id": 3,
        }

        # Add special tokens if thinking model
        if metadata.has_thinking:
            gguf_metadata["general.has_thinking"] = "true"
            thinking_tokens = {k: v for k, v in metadata.special_tokens.items()
                             if "thinking" in v.lower()}
            gguf_metadata["tokenizer.special_tokens"] = json.dumps(thinking_tokens)

        return gguf_metadata

    def create_conversion_config(self, model_name: str, output_path: Path):
        """Create conversion configuration file"""
        if model_name not in ZEN_METADATA:
            return

        metadata = ZEN_METADATA[model_name]

        conversion_config = {
            "model_name": model_name,
            "metadata": asdict(metadata),
            "conversion_settings": {
                "context_size": metadata.context_length,
                "vocab_type": "bpe",
                "use_f16": True,
                "quantization_options": {
                    "mobile": ["Q4_K_S", "Q4_K_M"],
                    "balanced": ["Q5_K_M", "Q4_K_M"],
                    "quality": ["Q6_K", "Q8_0"],
                    "server": ["Q8_0", "FP16"]
                }
            },
            "gguf_metadata": self.generate_gguf_metadata(model_name)
        }

        config_file = output_path / f"{model_name}_conversion_config.json"
        with open(config_file, 'w') as f:
            json.dump(conversion_config, f, indent=2)

        logger.info(f"Created conversion config: {config_file}")

    def preserve_all_metadata(self):
        """Preserve metadata for all Zen models"""
        logger.info("Preserving metadata for all Zen models...")

        for model_name in ZEN_METADATA.keys():
            logger.info(f"\nProcessing {model_name}...")

            # Create model-specific directory
            model_dir = self.output_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            # Generate all metadata files
            self.create_tokenizer_config(model_name, model_dir)
            self.create_model_card(model_name, model_dir)
            self.create_conversion_config(model_name, model_dir)

        # Create master metadata file
        master_metadata = {
            "zen_family": "Hanzo AI Zen Models",
            "version": "1.0.0",
            "models": list(ZEN_METADATA.keys()),
            "metadata": {name: asdict(meta) for name, meta in ZEN_METADATA.items()}
        }

        master_file = self.output_dir / "zen_models_metadata.json"
        with open(master_file, 'w') as f:
            json.dump(master_metadata, f, indent=2)

        logger.info(f"\nMaster metadata saved to {master_file}")


def main():
    preserver = MetadataPreserver()
    preserver.preserve_all_metadata()

    # Show summary
    logger.info("\nMetadata preservation complete!")
    logger.info("Generated files:")
    logger.info("- Tokenizer configurations")
    logger.info("- Model cards")
    logger.info("- Conversion configurations")
    logger.info("- Master metadata file")


if __name__ == "__main__":
    main()