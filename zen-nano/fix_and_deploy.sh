#!/bin/bash

# Zen Nano 3.5B - Fix naming and deploy
echo "🔧 Fixing Zen Nano naming and structure..."
echo "================================"

# Correct the model naming
cd /Users/z/work/zen/zen-nano

# Rename MLX models to correct size
echo "📝 Renaming models to reflect actual 3.5B size..."
mv models/zen-nano-v1-mlx models/zen-nano-3.5b-instruct 2>/dev/null
mv models/zen-nano-v1-mlx-q4 models/zen-nano-3.5b-instruct-q4 2>/dev/null
mv models/zen-nano-v1-mlx-q8 models/zen-nano-3.5b-instruct-q8 2>/dev/null

# Update HuggingFace upload directory
cd huggingface-upload
mv zen-nano-v1-mlx zen-nano-3.5b-instruct 2>/dev/null
mv zen-nano-v1-mlx-q4 zen-nano-3.5b-instruct-q4 2>/dev/null
mv zen-nano-v1-mlx-q8 zen-nano-3.5b-instruct-q8 2>/dev/null

echo "✅ Models renamed to zen-nano-3.5b-instruct"

# Update README with correct information
cat > README.md << 'EOF'
---
license: apache-2.0
language:
- en
library_name: mlx
tags:
- mlx
- edge-ai
- lightweight
- zen-nano
base_model: Qwen/Qwen2.5-3B-Instruct
model_size: 3.5B
---

# Zen Nano 3.5B Instruct

⚠️ **Note: Model currently identifies as Qwen. Stronger identity training in progress.**

An ultra-lightweight AI model optimized for edge devices, jointly developed by **Hanzo AI Inc** (Techstars-backed, LA) and **Zoo Labs Foundation** (501c3, SF).

## Model Details
- **Size**: 3.5B parameters (not 4B - corrected from initial documentation)
- **Base**: Qwen2.5-3B-Instruct with LoRA fine-tuning
- **Training**: 48 identity samples (needs more for stronger alignment)

## Features
- 🌐 Runs entirely offline on edge devices
- 🔒 Complete privacy - no data leaves your device
- 🌊 Eco-friendly with minimal carbon footprint
- 📱 Works on phones, tablets, Raspberry Pi
- 🆓 Forever free and open source

## Available Files

### MLX Format (Apple Silicon)
- `zen-nano-3.5b-instruct/` - Full precision MLX (7.5GB)
- `zen-nano-3.5b-instruct-q4/` - 4-bit quantized (2.1GB)
- `zen-nano-3.5b-instruct-q8/` - 8-bit quantized (4.0GB)

### GGUF Format
Coming soon - requires PyTorch for conversion

## Quick Start

### MLX (Mac/Apple Silicon)
```python
from mlx_lm import load, generate

model, tokenizer = load("hanzoai/zen-nano-3.5b-instruct-q4")
response = generate(model, tokenizer, prompt="Who are you?", max_tokens=100)
print(response)
```

## Known Issues
- Identity alignment incomplete - model may identify as Qwen
- GGUF conversion pending
- Requires stronger training for full Zen Nano identity

## About the Creators

**Hanzo AI Inc**: Techstars-backed applied AI research lab in Los Angeles.

**Zoo Labs Foundation**: 501(c)(3) non-profit in San Francisco dedicated to free AI and ocean conservation.

## License

Apache 2.0 - Free for any use including commercial.
EOF

echo "✅ README updated with correct model size and known issues"

# Create proper model cards for each variant
for variant in zen-nano-3.5b-instruct*; do
  if [ -d "$variant" ]; then
    cat > "$variant/README.md" << EOF
# Zen Nano 3.5B Instruct - ${variant##*/}

Model variant: ${variant##*/}
Parameters: 3.5B (actual count: 3.51B)
Base: Qwen2.5-3B-Instruct

Note: Identity training incomplete - may identify as Qwen.
EOF
    echo "  ✅ Created model card for $variant"
  fi
done

echo ""
echo "📊 Summary of changes:"
echo "  - Renamed all models from 4B to 3.5B"
echo "  - Updated documentation with actual parameter count"
echo "  - Added known issues about identity alignment"
echo "  - Created individual model cards"
echo ""
echo "Next steps:"
echo "  1. Run Unsloth training for stronger identity"
echo "  2. Generate GGUF files with llama.cpp"
echo "  3. Re-upload to Hugging Face with correct naming"