#!/bin/bash

# Zen MLX Setup Script
# Sets up MLX environment and downloads foundational models

echo "=== Zen MLX Setup ==="
echo

# Check if on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  Warning: This script is optimized for Apple Silicon (M1/M2/M3)"
    echo "   Detected architecture: $(uname -m)"
    echo
fi

# Create directories
echo "1. Creating directories..."
mkdir -p models
mkdir -p checkpoints
mkdir -p data

# Install Python dependencies
echo
echo "2. Installing MLX and dependencies..."
pip install -r requirements.txt

# Download models from Hugging Face
echo
echo "3. Downloading foundational models..."
echo "   This will download quantized MLX versions optimized for Apple Silicon"
echo

# Function to download model
download_model() {
    local model_name=$1
    local model_path=$2
    local model_dir=$3

    echo "   Downloading $model_name..."
    if [ -d "models/$model_dir" ]; then
        echo "   ✓ $model_name already exists"
    else
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$model_path',
    local_dir='models/$model_dir',
    local_dir_use_symlinks=False
)
print('   ✓ Downloaded $model_name')
"
    fi
}

# Download Qwen3 models
download_model "Qwen3-4B-Instruct" "mlx-community/Qwen3-4B-Instruct-2507-4bit" "qwen3-4b-instruct"
download_model "Qwen3-4B-Thinking" "mlx-community/Qwen3-4B-Thinking-2507-4bit" "qwen3-4b-thinking"

# Optional: Download additional models
read -p "Download additional models (Qwen2.5-7B)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    download_model "Qwen2.5-7B-Instruct" "mlx-community/Qwen2.5-7B-Instruct-4bit" "qwen2.5-7b"
fi

# Create sample fine-tuning data
echo
echo "4. Creating sample fine-tuning data..."
cat > data/sample_finetune.jsonl << 'EOF'
{"prompt": "What is Zen MLX?", "completion": "Zen MLX is a high-performance inference framework for running large language models on Apple Silicon using the MLX library."}
{"prompt": "How fast is MLX inference?", "completion": "MLX inference on Apple Silicon can achieve speeds of 50-100 tokens per second for 4B parameter models."}
{"prompt": "What models does Zen support?", "completion": "Zen supports all Qwen family models including Qwen3-4B, Qwen2.5-7B, and specialized models for coding and reasoning."}
EOF

echo "   ✓ Created sample_finetune.jsonl"

# Test installation
echo
echo "5. Testing MLX installation..."
python -c "
import mlx
import mlx_lm
print(f'   ✓ MLX version: {mlx.__version__}')
print(f'   ✓ MLX-LM installed')
print(f'   ✓ Available memory: {mlx.metal.get_active_memory() / 1e9:.1f} GB')
"

echo
echo "=== Setup Complete! ==="
echo
echo "To start the server:"
echo "  python mlx_server.py --model qwen3-4b-instruct --port 3690"
echo
echo "To fine-tune a model:"
echo "  python finetune.py --base-model models/qwen3-4b-instruct --data data/sample_finetune.jsonl"
echo
echo "Available models in models/:"
ls -la models/ 2>/dev/null | grep "^d" | awk '{print "  - " $NF}' | grep -v "^\.$\|^\.\.$" || echo "  (No models downloaded yet)"