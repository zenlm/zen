#!/bin/bash

# Zen-1 Setup Script
echo "==================================="
echo "       Zen-1 Model Setup"
echo "==================================="
echo

# Create remaining directories
echo "Setting up directory structure..."
mkdir -p instruct/{checkpoints,eval,data}
mkdir -p thinking/{checkpoints,eval,data}

# Check Python version
echo "Checking Python version..."
if command -v python3.11 &> /dev/null; then
    echo "✓ Python 3.11 found"
    PYTHON=python3.11
else
    echo "⚠ Python 3.11 not found, using default python3"
    PYTHON=python3
fi

# Install dependencies
echo
echo "Installing dependencies..."
$PYTHON -m pip install mlx mlx-lm pyyaml numpy

# Make scripts executable
chmod +x instruct/scripts/finetune.py
chmod +x thinking/scripts/finetune.py
chmod +x inference.py

echo
echo "==================================="
echo "      Setup Complete!"
echo "==================================="
echo
echo "Quick Start Commands:"
echo
echo "1. Test Inference (Instruct):"
echo "   ./inference.py 'Write a hello world in Python' --variant instruct"
echo
echo "2. Test Inference (Thinking):"
echo "   ./inference.py 'What is 25 * 37?' --variant thinking --show-thinking"
echo
echo "3. Interactive Mode:"
echo "   ./inference.py --interactive --variant instruct"
echo
echo "4. Fine-tune Instruct:"
echo "   cd instruct && python scripts/finetune.py"
echo
echo "5. Fine-tune Thinking:"
echo "   cd thinking && python scripts/finetune.py"
echo
echo "Training data samples available in:"
echo "  - instruct/data/instruct_train.jsonl"
echo "  - thinking/data/thinking_train.jsonl"