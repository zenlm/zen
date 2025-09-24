#!/bin/bash
# GGUF Conversion Script for Qwen3-Omni-30B-A3B-Thinking

MODEL_DIR="/Users/z/work/zen/qwen3-omni-30b-a3b-thinking"
OUTPUT_DIR="/Users/z/work/zen/qwen3-omni-30b-a3b-thinking-gguf"

echo "🔄 Converting to GGUF format..."

# Clone llama.cpp if not present
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp
make clean && make

# Convert to GGUF
python convert.py "$MODEL_DIR" --outtype q4_K_M --outfile "$OUTPUT_DIR/qwen3-omni-30b-q4.gguf"

echo "✅ GGUF conversion complete!"
