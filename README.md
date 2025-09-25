# Zen AI Model Ecosystem

A comprehensive collection of efficient AI models optimized for edge deployment.

## Models

### Zen Nano (4B Parameters)
- **zen-nano-thinking**: Advanced reasoning capabilities with chain-of-thought
- **zen-nano-instruct**: Optimized for instruction following
- Location: `models/zen-nano/`

### Zen Omni (30B Parameters)
- **zen-omni-thinking**: Large-scale reasoning model
- **zen-omni-instruct**: Enterprise-grade instruction model
- **zen-omni-captioner**: Specialized for image captioning
- Location: `models/zen-omni/`

### Zen Coder
- Specialized model for code generation and understanding
- Location: `models/zen-coder/`

### Zen Next
- Next-generation experimental models
- Location: `models/zen-next/`

### Zen 3D
- 3D understanding and generation capabilities
- Location: `models/zen-3d/`

## Directory Structure

```
zen/
├── models/           # All model implementations
├── training/         # Training scripts and configs
├── tools/           # Conversion and deployment tools
├── docs/            # Documentation and papers
├── examples/        # Usage examples
├── output/          # Generated outputs
└── external/        # External dependencies
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train a model:
```bash
python training/scripts/train_zen_nano.py
```

3. Convert to GGUF:
```bash
python tools/conversion/convert_to_gguf.py
```

4. Deploy to Hugging Face:
```bash
python tools/deployment/deploy_zen.py
```

## License

MIT License - See LICENSE file for details.

## Citation

```bibtex
@misc{zen2025,
  title={Zen: Efficient Edge AI Models},
  author={Zen Team},
  year={2025},
  publisher={GitHub},
  url={https://github.com/zenlm/zen}
}
```
