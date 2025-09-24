# Qwen3-Omni-MoE Runbook

## Quick Start

### 1. Local Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/zen.git
cd zen

# Install dependencies
pip install transformers torch peft datasets accelerate bitsandbytes

# Run the setup
python setup_zen_omni.py
```

### 2. Fine-tuning Qwen3-Omni
```bash
# Fine-tune with your data
python run_finetune.py

# Or use the Qwen3-Omni specific script
python use_real_qwen3.py
```

### 3. Using the Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load from HuggingFace
model = AutoModelForCausalLM.from_pretrained("zeekay/zen-qwen3-omni-moe")
tokenizer = AutoTokenizer.from_pretrained("zeekay/zen-qwen3-omni-moe")

# Or load locally
model = AutoModelForCausalLM.from_pretrained("./qwen3-omni-moe-final")
tokenizer = AutoTokenizer.from_pretrained("./qwen3-omni-moe-final")
```

## Architecture

### Thinker-Talker MoE Design
The Qwen3-Omni architecture uses a dual-module approach:
- **Thinker**: Processes multimodal inputs and reasons about them
- **Talker**: Generates responses with ultra-low latency

### Key Specifications
- Total Parameters: 30B (demo version is smaller)
- Active Parameters: 3B per forward pass
- Experts: 8 total, 2 active per token
- First-packet latency: 234ms target
- Languages: 119 text, 19 speech input, 10 speech output

## Training Configuration

### LoRA Settings
```python
lora_config = LoraConfig(
    r=4,  # Rank optimized for Apple Silicon
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
```

### MoE Configuration
```python
moe_config = {
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "expert_router": "top_k",
    "aux_loss_coef": 0.01
}
```

## Deployment

### Local Deployment
```bash
# Using Ollama
ollama create qwen3-omni -f Modelfile-qwen3-omni
ollama run qwen3-omni

# Using Python
python serve_model.py --model qwen3-omni-moe-final --port 8080
```

### Cloud Deployment
```bash
# Deploy to HuggingFace Spaces
python push_qwen3_omni_to_hf.py

# Deploy to Replicate
cog push r8.im/yourusername/qwen3-omni
```

## Performance Optimization

### Apple Silicon (M1/M2/M3)
- Use MPS device for acceleration
- Set `fp16=False` to avoid MPS mixed precision issues
- Use smaller batch sizes (1-2)
- Leverage unified memory architecture

### Memory Management
```python
# Clear cache periodically
import torch
torch.mps.empty_cache() if torch.backends.mps.is_available() else None

# Use gradient checkpointing
model.gradient_checkpointing_enable()
```

## Multimodal Capabilities

### Text Processing
```python
# Text-only input
response = model.generate(
    tokenizer("Explain quantum computing", return_tensors="pt").input_ids
)
```

### Image Understanding (Coming Soon)
```python
# Process image with text
from PIL import Image
image = Image.open("example.jpg")
response = processor.process_multimodal({"text": "What's in this image?", "image": image})
```

### Audio Processing (Coming Soon)
```python
# Process audio input
import librosa
audio, sr = librosa.load("audio.wav", sr=16000)
response = processor.process_audio(audio, sample_rate=sr)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use smaller LoRA rank
   - Enable gradient checkpointing

2. **MPS Errors**
   - Set `fp16=False`
   - Update PyTorch to latest version
   - Use CPU fallback if needed

3. **Import Errors**
   - Ensure transformers is up to date
   - Check PYTHONPATH includes project directory

## API Reference

### ZenOmniProcessor
```python
processor = ZenOmniProcessor(config)
processor.process_text(text)
processor.process_multimodal(inputs)
```

### ThinkerTalker
```python
model = ThinkerTalker(config)
thoughts = model.think(inputs)
response = model.talk(thoughts, stream=True)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

Apache 2.0

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/zen/issues
- Discord: Join our community server
- Documentation: https://zen-omni.readthedocs.io