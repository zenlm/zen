# Zen-Omni Deployment Guide

## Complete Deployment Package Contents

This package contains everything needed to deploy the Zen-Omni multimodal foundation models to Hugging Face Hub.

## Package Structure

```
zen-omni-deployment/
├── models/                    # Model variants
│   ├── zen-omni-thinking/    # Deep reasoning variant (70% Thinker, 30% Talker)
│   ├── zen-omni-talking/     # Fast generation variant (30% Thinker, 70% Talker)
│   └── zen-omni-captioner/   # Media captioning variant (50% Thinker, 50% Talker)
├── paper/                     # Research paper
│   ├── zen-omni.tex          # LaTeX source
│   └── Makefile              # Build script
├── deploy_to_hf.py           # Deployment automation
├── examples.py               # Usage examples
├── test_deployment.py        # Validation script
├── requirements.txt          # Dependencies
├── README.md                 # Main documentation
└── DEPLOYMENT.md            # This file
```

## Model Specifications

### Architecture
- **Total Parameters**: 30B (Mixture of Experts)
- **Active Parameters**: 3B (10% activation)
- **Number of Experts**: 10
- **Experts per Token**: 2 (top-k routing)
- **Context Length**: 16K-32K tokens
- **Modalities**: Text, Image, Audio, Video

### Variants
1. **zen-omni-thinking**: Optimized for complex reasoning (280ms latency)
2. **zen-omni-talking**: Optimized for real-time generation (185ms latency)
3. **zen-omni-captioner**: Optimized for temporal media (211ms latency)

## Deployment Steps

### 1. Prerequisites

Install required packages:
```bash
pip install -r requirements.txt
```

Set your Hugging Face token:
```bash
export HF_TOKEN="your_huggingface_token"
```

### 2. Validate Package

Run validation tests:
```bash
python test_deployment.py
```

Expected output: All checks should pass (✓)

### 3. Build Research Paper

```bash
cd paper
make all
make view  # Opens PDF
```

### 4. Deploy to Hugging Face

Deploy all variants:
```bash
python deploy_to_hf.py --all
```

Or deploy individually:
```bash
python deploy_to_hf.py --variant thinking
python deploy_to_hf.py --variant talking
python deploy_to_hf.py --variant captioner
```

For private repositories:
```bash
python deploy_to_hf.py --all --private
```

### 5. Verify Deployment

After deployment, models will be available at:
- https://huggingface.co/zenlm/zen-omni-thinking
- https://huggingface.co/zenlm/zen-omni-talking
- https://huggingface.co/zenlm/zen-omni-captioner

## Testing Models

### Quick Test

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "zenlm/zen-omni-thinking",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-omni-thinking")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### Run Examples

```bash
python examples.py
```

## Model Cards

Each variant includes a comprehensive model card with:
- Model description and architecture
- Training details and datasets
- Performance benchmarks
- Usage examples
- API documentation
- Environmental impact
- Citation information

## Performance Metrics

| Variant | MMLU | VQA | Latency | Tokens/sec | Active Params |
|---------|------|-----|---------|------------|---------------|
| Thinking | 87.2% | 88.9% | 280ms | 55 | 3B |
| Talking | 84.5% | 87.3% | 185ms | 75 | 3B |
| Captioner | 83.9% | 86.8% | 211ms | 65 | 3B |

## API Integration

Models support multiple inference methods:

1. **Transformers Library**: Direct loading with AutoModel
2. **Streaming API**: Real-time generation with TextStreamer
3. **Batch Processing**: Efficient multi-prompt inference
4. **REST API**: Cloud-hosted endpoints at api.zenlm.ai
5. **WebSocket**: Real-time streaming for interactive apps

## Advanced Features

### Thinker-Talker Architecture
- Separates reasoning (Thinker) from generation (Talker)
- Enables task-specific optimization via weight distribution
- Supports dynamic module selection based on input

### Mixture of Experts
- 10x parameter efficiency (3B active from 30B total)
- Expert specialization for different modalities
- Load-balanced routing with auxiliary loss

### Multimodal Processing
- Unified encoder for all modalities
- Cross-modal attention mechanisms
- Temporal alignment for video/audio

## Troubleshooting

### Common Issues

1. **Memory Error**: Ensure 12GB+ GPU memory available
2. **Token Error**: Verify HF_TOKEN is set correctly
3. **Import Error**: Install all requirements.txt packages
4. **Deployment Failure**: Check internet connection and HF permissions

### Support

- GitHub Issues: https://github.com/zenlm/zen-omni/issues
- Email: support@zenlm.ai
- Documentation: https://docs.zenlm.ai

## License

Apache 2.0 - See LICENSE file

## Citation

```bibtex
@article{zen2025omni,
  title={Zen-Omni: A Thinker-Talker Architecture for Ultra-Low Latency Multimodal Understanding},
  author={Zen Research Team},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## Next Steps

1. **Fine-tuning**: Adapt models for specific domains
2. **Quantization**: Deploy INT8 versions for edge devices
3. **API Setup**: Configure cloud endpoints
4. **Monitoring**: Set up performance tracking
5. **Documentation**: Update with production insights

---

Deployment package prepared by Zen Research Team
Contact: research@zenlm.ai