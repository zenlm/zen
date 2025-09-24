#!/bin/bash

echo "╔═══════════════════════════════════════════════════════╗"
echo "║         DEPLOYING ZEN-OMNI MULTIMODAL MODELS         ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install from https://ollama.ai"
    exit 1
fi

echo "🚀 Creating Zen-Omni models..."

# Create each variant
for variant in instruct thinking captioner; do
    echo
    echo "📦 Building zen-omni-$variant..."
    
    if [ -f "Modelfile.zen-omni-$variant" ]; then
        ollama create zen-omni-$variant -f Modelfile.zen-omni-$variant
        echo "✅ Created zen-omni-$variant"
    else
        echo "❌ Modelfile.zen-omni-$variant not found"
    fi
done

echo
echo "🧪 Testing Zen-Omni models..."
echo

# Test each model
echo "Testing zen-omni-instruct:"
echo "How do I use hanzo-mcp for multimodal processing?" | ollama run zen-omni-instruct --verbose=false 2>/dev/null | head -5

echo
echo "═══════════════════════════════════════════════════════"
echo "✅ Zen-Omni Deployment Complete!"
echo "═══════════════════════════════════════════════════════"
echo
echo "Available models:"
ollama list | grep zen-omni

echo
echo "Usage examples:"
echo "  ollama run zen-omni-instruct"
echo "  ollama run zen-omni-thinking"
echo "  ollama run zen-omni-captioner"
echo
echo "API usage:"
echo "  curl http://localhost:11434/api/generate \"
echo "    -d '{"model": "zen-omni-instruct", "prompt": "Analyze this", "stream": true}'"
