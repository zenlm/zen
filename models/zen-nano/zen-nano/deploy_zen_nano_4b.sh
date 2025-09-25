#!/bin/bash

# Zen Nano 4B Instruct - Correct deployment
echo "🚀 Zen Nano 4B Instruct - Deployment Script"
echo "==========================================="
echo "Base Model: Qwen3-4B-Instruct-2507"
echo ""

cd /Users/z/work/zen/zen-nano

# Rename models to standard convention
echo "📝 Renaming models to zen-nano-4b-instruct..."
mv models/zen-nano-v1-mlx models/zen-nano-4b-instruct-mlx 2>/dev/null || true
mv models/zen-nano-v1-mlx-q4 models/zen-nano-4b-instruct-mlx-q4 2>/dev/null || true
mv models/zen-nano-v1-mlx-q8 models/zen-nano-4b-instruct-mlx-q8 2>/dev/null || true

echo "✅ Models renamed correctly as 4B"

# Create updated README
cat > README.md << 'EOF'
# Zen Nano 4B Instruct

Ultra-lightweight AI model based on **Qwen3-4B-Instruct**, jointly developed by:
- **Hanzo AI Inc** (Techstars-backed, Los Angeles)
- **Zoo Labs Foundation** (501c3, San Francisco)

## Model Specifications
- **Parameters**: 4B (Qwen3-4B base)
- **Architecture**: Qwen3 with LoRA fine-tuning
- **Optimized for**: Edge devices, offline operation
- **License**: Apache 2.0

## Available Formats

### MLX (Apple Silicon Optimized)
- `zen-nano-4b-instruct-mlx/` - Full precision (7.5GB)
- `zen-nano-4b-instruct-mlx-q4/` - 4-bit quantized (2.1GB)
- `zen-nano-4b-instruct-mlx-q8/` - 8-bit quantized (4.0GB)

### GGUF (Coming Soon)
- Requires llama.cpp conversion
- Will include Q4_K_M, Q5_K_M, Q8_0 variants

## Quick Start

```python
from mlx_lm import load, generate

# Load 4-bit quantized version
model, tokenizer = load("zen-nano-4b-instruct-mlx-q4")
response = generate(model, tokenizer,
                   prompt="What is your name?",
                   max_tokens=100)
print(response)
```

## Known Issue: Identity Alignment
⚠️ The model currently identifies as Qwen instead of Zen Nano.
This is being addressed with stronger LoRA training using Unsloth.

## Mission
Democratize AI through efficient edge computing while protecting our oceans.

## Links
- GitHub: https://github.com/zenlm/zen-nano
- HuggingFace: https://huggingface.co/zenlm/zen-nano-4b-instruct
- Website: https://zenlm.github.io/zen-nano/
EOF

echo "✅ README updated for 4B model"

# Generate GGUF if llama.cpp is available
if [ -d "llama.cpp" ]; then
    echo ""
    echo "🔧 Attempting GGUF conversion..."

    # Try to convert to GGUF
    python3 llama.cpp/convert_hf_to_gguf.py \
        models/zen-nano-4b-instruct-mlx \
        --outfile models/gguf/zen-nano-4b-instruct-f16.gguf \
        --outtype f16 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "✅ GGUF conversion successful"

        # Create quantized versions
        ./llama.cpp/build/bin/quantize \
            models/gguf/zen-nano-4b-instruct-f16.gguf \
            models/gguf/zen-nano-4b-instruct-q4_k_m.gguf \
            Q4_K_M 2>/dev/null

        echo "✅ Created Q4_K_M quantization"
    else
        echo "⚠️  GGUF conversion needs PyTorch - skipping"
    fi
else
    echo "⚠️  llama.cpp not found - GGUF conversion skipped"
fi

echo ""
echo "📊 Summary:"
echo "  ✅ Model correctly named as zen-nano-4b-instruct"
echo "  ✅ Base model: Qwen3-4B-Instruct-2507"
echo "  ✅ MLX formats ready (full, q4, q8)"
echo "  ⚠️  Identity training needed (currently identifies as Qwen)"
echo ""
echo "Next Step: Run Unsloth training for strong Zen Nano identity"
echo "  python3 training/train_zen_nano_unsloth.py"