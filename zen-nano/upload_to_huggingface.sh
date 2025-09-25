#!/bin/bash

# Zen Nano v1.0 - Upload to Hugging Face
# Jointly developed by Hanzo AI Inc and Zoo Labs Foundation

echo "🚀 Zen Nano v1.0 - Hugging Face Upload Script"
echo "============================================"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Installing huggingface-hub..."
    pip install huggingface-hub
fi

cd huggingface-upload

# Initialize git if needed
if [ ! -d ".git" ]; then
    git init
fi

# Add all files
echo "📁 Adding files to git..."
git add -A

# Commit
echo "💾 Creating commit..."
git commit -m "Zen Nano v1.0 - Ultra-lightweight edge AI model

Jointly developed by:
- Hanzo AI Inc (Techstars-backed, Los Angeles)
- Zoo Labs Foundation (501c3, San Francisco)

Features:
- MLX format for Apple Silicon (full, 4-bit, 8-bit)
- Runs entirely offline on edge devices
- Minimal carbon footprint
- Free and open source (Apache 2.0)"

echo ""
echo "📤 Ready to upload!"
echo ""
echo "To complete upload:"
echo "1. Login: huggingface-cli login"
echo "2. Create repo: huggingface-cli repo create zen-nano-v1 --type model --organization hanzo-ai"
echo "3. Add remote: git remote add origin https://huggingface.co/hanzo-ai/zen-nano-v1"
echo "4. Push: git push -u origin main"
echo ""
echo "Or use the HF web interface to create the repo first, then push."