#!/bin/bash
# Complete Qwen3-Omni MLX 4-bit and LM Studio Setup

echo "=================================================="
echo "Qwen3-Omni-30B MLX 4-bit Setup for LM Studio"
echo "=================================================="

MODEL_PATH="$HOME/work/zen/qwen3-omni-30b-complete"
MLX_PATH="$HOME/work/zen/qwen3-omni-mlx"
GGUF_PATH="$HOME/work/zen/qwen3-omni-gguf"

# Step 1: Convert to MLX format
echo ""
echo "Step 1: Convert Qwen3-Omni to MLX"
echo "----------------------------------"
echo "python3 -m mlx_lm.convert \\"
echo "  --hf-path $MODEL_PATH \\"
echo "  --mlx-path $MLX_PATH/fp16 \\"
echo "  --dtype float16"

# Step 2: Quantize to 4-bit  
echo ""
echo "Step 2: Quantize to 4-bit (reduces from 60GB to ~15GB)"
echo "-------------------------------------------------------"
echo "python3 -m mlx_lm.quantize \\"
echo "  --model $MLX_PATH/fp16 \\"
echo "  --bits 4 \\"
echo "  --output-dir $MLX_PATH/q4"

# Step 3: Test MLX model
echo ""
echo "Step 3: Test MLX Model"
echo "----------------------"
cat << 'EOF' > test_mlx.py
from mlx_lm import load, generate
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "~/work/zen/qwen3-omni-mlx/q4"
print(f"Loading model from {model_path}...")

model, tokenizer = load(model_path)
print("Model loaded successfully!")

prompt = "You are Zen-Omni, a powerful multimodal AI. Explain your capabilities:"
print(f"\nPrompt: {prompt}")

response = generate(
    model, 
    tokenizer,
    prompt=prompt,
    max_tokens=100,
    temperature=0.7,
    verbose=True
)

print(f"\nResponse: {response}")
EOF

echo "python3 test_mlx.py $MLX_PATH/q4"

# Step 4: Convert to GGUF for LM Studio
echo ""
echo "Step 4: Convert to GGUF for LM Studio"
echo "--------------------------------------"
echo ""
echo "Option A: Using llama.cpp converter"
echo "# Install llama.cpp"
echo "git clone https://github.com/ggerganov/llama.cpp"
echo "cd llama.cpp && make"
echo ""
echo "# Convert to GGUF"
echo "python3 llama.cpp/convert_hf_to_gguf.py $MODEL_PATH \\"
echo "  --outfile $GGUF_PATH/qwen3-omni-30b.gguf \\"
echo "  --outtype f16"
echo ""
echo "# Quantize to 4-bit"
echo "llama.cpp/llama-quantize $GGUF_PATH/qwen3-omni-30b.gguf \\"
echo "  $GGUF_PATH/qwen3-omni-30b-q4_k_m.gguf Q4_K_M"

echo ""
echo "Option B: Direct conversion (if supported)"
echo "python3 -c \"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_PATH',
    torch_dtype=torch.float16,
    device_map='cpu'
)

print('Saving in GGUF format...')
# Note: This requires GGUF export support
model.save_pretrained('$GGUF_PATH', safe_serialization=True, max_shard_size='2GB')
\""

# Step 5: LM Studio Setup
echo ""
echo "Step 5: Load in LM Studio"
echo "-------------------------"
echo "1. Copy GGUF to LM Studio:"
echo "   cp $GGUF_PATH/qwen3-omni-30b-q4_k_m.gguf \\"
echo "     ~/Library/Application\ Support/LM\ Studio/models/"
echo ""
echo "2. LM Studio Settings:"
echo "   - Model: qwen3-omni-30b-q4_k_m.gguf"
echo "   - Context: 8192"
echo "   - GPU Layers: -1 (use all)"
echo "   - Temperature: 0.7"
echo "   - System Prompt: 'You are Zen-Omni, a hypermodal AI assistant.'"
echo ""

# Performance expectations
echo "=================================================="
echo "Performance Expectations"
echo "=================================================="
echo ""
echo "MLX 4-bit on Apple Silicon:"
echo "  - Model size: ~15GB (from 60GB)"
echo "  - RAM required: 20-24GB"
echo "  - M1 Max: ~8-12 tokens/sec"
echo "  - M2 Ultra: ~20-25 tokens/sec"
echo "  - M3 Max: ~15-20 tokens/sec"
echo ""
echo "GGUF 4-bit in LM Studio:"
echo "  - Model size: ~15GB"
echo "  - RAM required: 18-20GB"
echo "  - Speed: Similar to MLX"
echo ""

# Status check
echo "=================================================="
echo "Current Status"
echo "=================================================="
echo ""
echo "Checking download status..."
if [ -f "$MODEL_PATH/model-00016-of-00016.safetensors" ]; then
    echo "✅ Model fully downloaded!"
    echo "Ready to convert to MLX"
else
    echo "⏳ Model still downloading..."
    echo "Check with: ls -la $MODEL_PATH/"
fi

echo ""
echo "To run the conversion pipeline:"
echo "1. Wait for download to complete"
echo "2. Run the MLX conversion commands above"
echo "3. Test with the Python script"
echo "4. Convert to GGUF for LM Studio"
echo "5. Load in LM Studio and enjoy!"