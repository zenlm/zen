# BitDelta: 1-Bit Personalization for Zen Models

BitDelta enables efficient personalization of large language models by storing only 1-bit representations of weight deltas, achieving 10-100x compression while maintaining quality.

## Features

- **Extreme Compression**: 100x smaller than full model copies
- **Fast Switching**: Change personalization profiles in milliseconds  
- **Multi-Profile Support**: Maintain hundreds of user profiles efficiently
- **Quality Retention**: 92-97% performance vs full fine-tuning
- **Privacy-Preserving**: Personal data never leaves device

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from zen_integration import ZenPersonalizationManager

# Initialize with Zen model
manager = ZenPersonalizationManager('zen-nano-4b')

# Create personalization from examples
manager.create_profile_from_examples(
    'technical',
    technical_examples,
    style='technical writing'
)

# Generate with personalization
response = manager.generate_with_profile(
    "Explain quantum computing:",
    profile_name='technical'
)
```

## Core Components

### `bitdelta.py`
Core BitDelta implementation with encoder/decoder and compression logic.

### `training.py` 
Training pipelines including progressive quantization and multi-profile management.

### `zen_integration.py`
Integration with Zen model family (nano, omni, coder variants).

### `example_usage.py`
Complete examples demonstrating personalization workflows.

## Compression Ratios

| Model | Size | BitDelta | Ratio |
|-------|------|----------|-------|
| zen-nano-4b | 16GB | 150MB | 107x |
| zen-next-7b | 28GB | 280MB | 100x |
| zen-omni-30b | 120GB | 1.2GB | 100x |

## Testing

```bash
python test_bitdelta.py
```

## Paper

See `zoo_paper.md` for technical details and experimental results.

## Architecture

BitDelta stores personalization as:
- **Sign bits**: 1 bit per parameter (direction of change)
- **Scale factors**: One float per layer/channel (magnitude)

Reconstruction: `W_personalized = W_base + α · sign(ΔW)`

This achieves massive compression while preserving essential personalization characteristics.