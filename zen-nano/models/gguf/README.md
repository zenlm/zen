# Zen Nano 4B GGUF Models

🦙 **Universal GGUF formats** for all Zen Nano 4B models, compatible with llama.cpp and any GGUF-supporting inference engine.

Jointly developed by [Hanzo AI Inc](https://hanzo.ai) (Techstars-backed, Los Angeles) and [Zoo Labs Foundation](https://zoolabs.org) (501c3, San Francisco).

## Available Models

### zen-nano-instruct
- **zen-nano-instruct-Q4_K_M.gguf**: 4-bit medium quality (~2.5GB)
- **zen-nano-instruct-Q5_K_M.gguf**: 5-bit medium quality (~3.1GB) 
- **zen-nano-instruct-Q6_K.gguf**: 6-bit quality (~3.7GB)
- **zen-nano-instruct-Q8_0.gguf**: 8-bit quality (~4.3GB)
- **zen-nano-instruct-f16.gguf**: Full precision (~8GB)

### zen-nano-instruct-4bit
- **zen-nano-instruct-4bit-Q4_K_M.gguf**: 4-bit medium quality (~2.5GB)
- **zen-nano-instruct-4bit-Q5_K_M.gguf**: 5-bit medium quality (~3.1GB)
- **zen-nano-instruct-4bit-Q6_K.gguf**: 6-bit quality (~3.7GB) 
- **zen-nano-instruct-4bit-Q8_0.gguf**: 8-bit quality (~4.3GB)

### zen-nano-thinking
- **zen-nano-thinking-Q4_K_M.gguf**: 4-bit medium quality (~2.5GB)
- **zen-nano-thinking-Q5_K_M.gguf**: 5-bit medium quality (~3.1GB)
- **zen-nano-thinking-Q6_K.gguf**: 6-bit quality (~3.7GB)
- **zen-nano-thinking-Q8_0.gguf**: 8-bit quality (~4.3GB)
- **zen-nano-thinking-f16.gguf**: Full precision (~8GB)

### zen-nano-thinking-4bit
- **zen-nano-thinking-4bit-Q4_K_M.gguf**: 4-bit medium quality (~2.5GB)
- **zen-nano-thinking-4bit-Q5_K_M.gguf**: 5-bit medium quality (~3.1GB)
- **zen-nano-thinking-4bit-Q6_K.gguf**: 6-bit quality (~3.7GB)
- **zen-nano-thinking-4bit-Q8_0.gguf**: 8-bit quality (~4.3GB)

## Usage

### llama.cpp

```bash
# Download model (example: instruct 4-bit)
wget https://huggingface.co/zenlm/zen-nano-instruct/resolve/main/zen-nano-instruct-Q4_K_M.gguf

# Run inference
./llama-cli -m zen-nano-instruct-Q4_K_M.gguf -p "User: What is your name?\nAssistant:" -n 50

# Interactive chat
./llama-cli -m zen-nano-instruct-Q4_K_M.gguf --interactive-first
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="zen-nano-instruct-Q4_K_M.gguf",
    n_ctx=2048,
    verbose=False
)

# Generate
output = llm(
    prompt="User: Explain quantum computing\nAssistant:",
    max_tokens=100,
    stop=["User:", "\n\n"]
)

print(output['choices'][0]['text'])
```

### Ollama

```bash
# Create Modelfile
echo 'FROM ./zen-nano-instruct-Q4_K_M.gguf' > Modelfile

# Import model
ollama create zen-nano -f Modelfile

# Run
ollama run zen-nano "What is your name?"
```

## Quality Recommendations

- **Q4_K_M**: Best balance of size/quality for most use cases
- **Q5_K_M**: Higher quality with moderate size increase
- **Q6_K**: Near-original quality for critical applications  
- **Q8_0**: Minimal quality loss, larger file size
- **f16**: Full precision, largest files

## Model Specifications

- **Architecture**: Qwen3 4B
- **Context Length**: 32,768 tokens
- **Vocabulary Size**: 151,936 tokens
- **Training**: Fine-tuned for identity alignment and ocean conservation mission
- **License**: Apache 2.0

## Performance Benchmarks

| Quantization | Size | Tokens/sec (CPU) | Quality Score |
|--------------|------|------------------|---------------|
| Q4_K_M       | 2.5GB| ~15-25           | 95%           |
| Q5_K_M       | 3.1GB| ~12-20           | 97%           |  
| Q6_K         | 3.7GB| ~10-18           | 98%           |
| Q8_0         | 4.3GB| ~8-15            | 99%           |
| f16          | 8.0GB| ~5-12            | 100%          |

*Benchmarks on Apple M1 Pro, single-threaded*

## Download Links

All models are available on Hugging Face:

- **[zen-nano-instruct](https://huggingface.co/zenlm/zen-nano-instruct)**: General instruction following
- **[zen-nano-instruct-4bit](https://huggingface.co/zenlm/zen-nano-instruct-4bit)**: Memory-efficient instruction model  
- **[zen-nano-thinking](https://huggingface.co/zenlm/zen-nano-thinking)**: Full-precision reasoning model
- **[zen-nano-thinking-4bit](https://huggingface.co/zenlm/zen-nano-thinking-4bit)**: Ultra-efficient reasoning model

## About the Creators

**Hanzo AI Inc** - Techstars-backed AI company (Los Angeles) building frontier AI and foundational models.

**Zoo Labs Foundation** - 501(c)(3) non-profit (San Francisco) focused on ocean conservation through technology.

---

*Universal GGUF formats bringing Zen AI to every platform while supporting ocean conservation.*
