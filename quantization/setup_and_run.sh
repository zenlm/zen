#!/bin/bash
# Setup and run 4-bit quantization for Zen models

set -e

echo "=== Zen 4-bit Quantization Setup ==="

# Install dependencies
echo "Installing Unsloth and dependencies..."
pip install -q -r requirements.txt

# Quantize models
echo -e "\n=== Quantizing Models ==="

# Zen-nano (4B)
echo -e "\n1. Quantizing zen-nano (4B)..."
python quantize_zen.py zen-nano --benchmark

# Zen-omni (30B) - optional due to size
echo -e "\n2. Zen-omni (30B) quantization:"
echo "   Run: python quantize_zen.py zen-omni --benchmark"
echo "   Note: Requires ~16GB GPU memory"

echo -e "\n=== Complete ==="
echo "Quantized models saved to ./quantized/"
