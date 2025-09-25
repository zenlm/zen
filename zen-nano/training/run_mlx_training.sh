#!/bin/bash
# Zen Nano 4B MLX Training Script

echo "🚀 Starting Zen Nano 4B Identity Training"
echo "Base: 8-bit MLX model from lmstudio-community"
echo "Training examples: 165"
echo ""

# Run MLX LoRA training
python3.12 -m mlx_lm lora \
    --model base-models/Qwen3-4B-Instruct-2507-MLX-8bit \
    --data training \
    --train \
    --batch-size 4 \
    --num-layers 16 \
    --iters 500 \
    --learning-rate 5e-5 \
    --warmup-steps 20 \
    --save-every 100 \
    --adapter-path models/zen-nano-4b-adapters \
    --test \
    --test-batches 10 \
    --val-batches 10

echo ""
echo "✅ Training complete!"
echo ""
echo "To test the model:"
echo "  python3.12 test_identity.py"
