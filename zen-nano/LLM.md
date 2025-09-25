# Zen Nano 4B Instruct - Comprehensive Deployment Guide

## Model Information
- **Name**: zen-nano-4b-instruct
- **Parameters**: 4B
- **Base Model**: Qwen3-4B-Instruct-2507
- **Created by**: Hanzo AI Inc & Zoo Labs Foundation
- **Version**: 1.0

## Table of Contents
1. [Model Format Conversion](#model-format-conversion)
2. [HuggingFace Publication](#huggingface-publication)
3. [Platform Integration](#platform-integration)
4. [Model Card Structure](#model-card-structure)
5. [Testing Procedures](#testing-procedures)
6. [Deployment Checklist](#deployment-checklist)

## Model Format Conversion

### 1. SafeTensors Format (Primary Distribution)

SafeTensors is the recommended format for HuggingFace distribution due to security and efficiency.

```bash
# Convert from GGUF to SafeTensors using llama.cpp
cd /Users/z/work/zen/zen-nano/llama.cpp

# First, convert GGUF to PyTorch format
python convert_gguf_to_hf.py \
    --input ../zen-nano.gguf \
    --output ../zen-nano-safetensors \
    --model-name zen-nano-4b-instruct

# Install required packages
pip install safetensors transformers torch

# Convert PyTorch to SafeTensors
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from safetensors.torch import save_file

model = AutoModelForCausalLM.from_pretrained('../zen-nano-safetensors')
tokenizer = AutoTokenizer.from_pretrained('../zen-nano-safetensors')

# Save model weights in safetensors format
state_dict = model.state_dict()
save_file(state_dict, '../zen-nano-safetensors/model.safetensors')

# Save tokenizer
tokenizer.save_pretrained('../zen-nano-safetensors')
"
```

### 2. GGUF Format (llama.cpp/Ollama)

GGUF is already available. For different quantization levels:

```bash
cd /Users/z/work/zen/zen-nano/llama.cpp

# Build quantization tool
make quantize

# Create different quantization levels
./quantize ../zen-nano.gguf ../zen-nano-q4_k_m.gguf q4_K_M  # Recommended balanced
./quantize ../zen-nano.gguf ../zen-nano-q5_k_m.gguf q5_K_M  # Higher quality
./quantize ../zen-nano.gguf ../zen-nano-q8_0.gguf q8_0      # Minimal loss
./quantize ../zen-nano.gguf ../zen-nano-q3_k_s.gguf q3_K_S  # Smaller size
```

### 3. MLX Format (Apple Silicon)

For optimal performance on Apple Silicon:

```bash
# Install MLX converter
pip install mlx mlx-lm

# Convert from SafeTensors to MLX
python -m mlx_lm.convert \
    --model ../zen-nano-safetensors \
    --output ../zen-nano-mlx \
    --quantize-bits 4  # Options: 4, 8, or 16

# Test the MLX model
python -m mlx_lm.generate \
    --model ../zen-nano-mlx \
    --prompt "You are Zen Nano" \
    --max-tokens 100
```

### 4. ONNX Format (Cross-platform)

```bash
pip install optimum[onnxruntime]

# Convert to ONNX
optimum-cli export onnx \
    --model ../zen-nano-safetensors \
    --task text-generation \
    ../zen-nano-onnx/

# Optimize ONNX model
python -m onnxruntime.tools.optimizer_cli \
    --input ../zen-nano-onnx/model.onnx \
    --output ../zen-nano-onnx/model_optimized.onnx \
    --model_type gpt2
```

## HuggingFace Publication

### Repository Structure

```
hanzoai/zen-nano-4b-instruct/
├── README.md                    # Model card
├── config.json                  # Model configuration
├── tokenizer_config.json        # Tokenizer configuration
├── tokenizer.json              # Tokenizer
├── special_tokens_map.json     # Special tokens
├── model.safetensors           # Main model weights
├── pytorch_model.bin           # PyTorch weights (optional)
├── gguf/                       # GGUF quantizations
│   ├── zen-nano-q4_k_m.gguf
│   ├── zen-nano-q5_k_m.gguf
│   └── zen-nano-q8_0.gguf
└── mlx/                        # MLX format
    ├── model.safetensors
    └── config.json
```

### Upload Process

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login to HuggingFace
huggingface-cli login

# Create repository
huggingface-cli repo create zen-nano-4b-instruct --organization hanzoai

# Clone the repository
git clone https://huggingface.co/hanzoai/zen-nano-4b-instruct
cd zen-nano-4b-instruct

# Copy model files
cp -r /Users/z/work/zen/zen-nano/zen-nano-safetensors/* .
mkdir -p gguf mlx
cp /Users/z/work/zen/zen-nano/*.gguf gguf/
cp -r /Users/z/work/zen/zen-nano/zen-nano-mlx/* mlx/

# Add LFS tracking for large files
git lfs track "*.bin"
git lfs track "*.safetensors"
git lfs track "*.gguf"
git lfs track "*.onnx"

# Commit and push
git add .
git commit -m "Initial release of Zen Nano 4B Instruct v1.0"
git push
```

## Platform Integration

### 1. Ollama Integration

Create Modelfile:

```dockerfile
# /Users/z/work/zen/zen-nano/Modelfile
FROM ./zen-nano-q4_k_m.gguf

TEMPLATE """{{ if .System }}System: {{ .System }}
{{ end }}User: {{ .Prompt }}
Assistant: """

SYSTEM """You are Zen Nano, an open AI model created by Hanzo AI for the Zoo Labs Foundation. You are part of the open AI ecosystem protecting wildlife and oceans."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
```

Deploy to Ollama:

```bash
# Create Ollama model
ollama create zen-nano:4b-instruct -f Modelfile

# Test the model
ollama run zen-nano:4b-instruct "Tell me about yourself"

# Push to Ollama registry (optional)
ollama push hanzoai/zen-nano:4b-instruct
```

### 2. llama.cpp Integration

```bash
cd /Users/z/work/zen/zen-nano/llama.cpp

# Build llama.cpp
make clean && make LLAMA_METAL=1  # For Apple Silicon
# OR
make clean && make LLAMA_CUDA=1   # For NVIDIA GPUs

# Run the model
./main -m ../zen-nano-q4_k_m.gguf \
    -p "You are Zen Nano" \
    -n 512 \
    -c 4096 \
    --temp 0.7 \
    --top-p 0.9 \
    --top-k 40

# Start server for API access
./server -m ../zen-nano-q4_k_m.gguf \
    -c 4096 \
    --host 0.0.0.0 \
    --port 8080
```

### 3. MLX Integration (Apple Silicon)

```python
# test_mlx.py
import mlx.core as mx
from mlx_lm import load, generate

# Load model
model, tokenizer = load("/Users/z/work/zen/zen-nano/zen-nano-mlx")

# Generate text
prompt = "You are Zen Nano, created by Hanzo AI. Your mission is"
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=100,
    temperature=0.7,
    top_p=0.9
)
print(response)
```

### 4. Transformers Library Integration

```python
# test_transformers.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "/Users/z/work/zen/zen-nano/zen-nano-safetensors"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # or torch.bfloat16
    device_map="auto"
)

# Set up generation config
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_new_tokens": 512,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

# Generate
prompt = "You are Zen Nano. Explain your mission:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, **generation_config)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Model Card Structure

Create a comprehensive `README.md` for HuggingFace with YAML frontmatter and detailed model documentation. The template should include:

- Proper YAML metadata tags for HuggingFace model hub
- License information (Apache 2.0)
- Model architecture details
- Training methodology
- Performance benchmarks
- Usage examples for all supported formats
- Citation information
- Links to Hanzo AI and Zoo Labs Foundation

Example structure available in the HuggingFace Model Card Best Practices guide.

## Testing Procedures

### 1. Identity Alignment Testing

Test that the model correctly identifies itself as Zen Nano:

```python
# test_identity.py
test_prompts = [
    "Who are you?",
    "What is your name?",
    "Who created you?",
    "What is your mission?",
    "Are you Qwen?"
]

expected_responses = {
    "identity": ["Zen Nano", "zen-nano"],
    "creators": ["Hanzo AI", "Zoo Labs"],
    "mission": ["conservation", "ocean", "wildlife"],
    "negative": ["not Qwen", "I am Zen Nano"]
}

# Run tests with low temperature for consistency
for prompt in test_prompts:
    response = generate(prompt, temperature=0.1)
    verify_keywords(response, expected_responses)
```

### 2. Platform Compatibility Testing

Test across all target platforms:

- **Ollama**: Create model, test generation, verify identity
- **llama.cpp**: Test CLI and server modes
- **MLX**: Verify Apple Silicon performance
- **Transformers**: Test PyTorch and SafeTensors loading
- **ONNX**: Verify cross-platform compatibility

### 3. Performance Benchmarking

Measure key metrics:

- Tokens per second across different formats
- Memory usage for each quantization level
- Perplexity scores for quality assessment
- Latency measurements for edge deployment

### 4. Quality Assurance

- Verify conservation knowledge accuracy
- Test instruction-following capabilities
- Validate safety guardrails
- Check for bias and fairness

## Deployment Checklist

### Pre-Deployment
- [ ] Model formats ready (GGUF, SafeTensors, MLX, ONNX)
- [ ] Quantization levels tested (Q3, Q4, Q5, Q8)
- [ ] Configuration files validated
- [ ] Identity alignment verified (>95% accuracy)
- [ ] Platform tests passed

### HuggingFace Publication
- [ ] Repository created with proper naming
- [ ] LFS configured for large files
- [ ] Model card complete with all sections
- [ ] Usage examples tested
- [ ] License and citations included

### Platform Integration
- [ ] Ollama Modelfile tested
- [ ] llama.cpp server verified
- [ ] MLX performance optimized
- [ ] Transformers compatibility confirmed
- [ ] API endpoints documented

### Post-Deployment
- [ ] Monitor download metrics
- [ ] Collect user feedback
- [ ] Track performance issues
- [ ] Update documentation
- [ ] Manage version releases

## Troubleshooting Guide

### Common Issues and Solutions

1. **GGUF Conversion Failures**
   - Update llama.cpp to latest version
   - Verify GGUF format compatibility
   - Check model architecture support

2. **Memory Issues**
   - Use appropriate quantization levels
   - Enable memory mapping for large models
   - Consider sharded loading for SafeTensors

3. **Identity Misalignment**
   - Increase LoRA rank if retraining
   - Use lower temperature for identity queries
   - Verify system prompt inclusion

4. **Performance Problems**
   - Choose optimal quantization for use case
   - Enable GPU acceleration where available
   - Use batch processing for throughput

## Support Resources

### Documentation
- [HuggingFace Model Hub](https://huggingface.co/docs/hub)
- [llama.cpp Guide](https://github.com/ggerganov/llama.cpp)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Ollama Docs](https://ollama.ai/docs)

### Contact
- **Hanzo AI**: contact@hanzo.ai
- **Zoo Labs**: info@zoolabs.org
- **GitHub Issues**: Report bugs and request features

### Community
- HuggingFace Forums
- llama.cpp Discord
- MLX Community
- Ollama Discord

---

*Document Version: 1.0*
*Last Updated: September 2024*
*Model: zen-nano-4b-instruct*
*Created by: Hanzo AI Inc & Zoo Labs Foundation*