#!/bin/bash
# Deploy Zen1-Omni Models

echo "╔═══════════════════════════════════════════════════════╗"
echo "║            ZEN1-OMNI DEPLOYMENT                      ║"
echo "║     The First Zen Multimodal AI Model                ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Check if model exists
if [ ! -d "./zen1-omni-branded" ]; then
    echo "⚠️  Zen1-Omni model not found. Running branding first..."
    python quick_zen1_brand.py
fi

echo "🚀 Deploying Zen1-Omni..."

# Create Modelfile for Ollama
cat > Modelfile.zen1-omni << 'EOF'
# Zen1-Omni: The First Zen Multimodal AI Model
FROM ./zen1-omni-branded

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER num_ctx 8192

SYSTEM """
You are Zen1-Omni, the first generation of Zen AI's revolutionary multimodal models. 

Key Identity:
- Model: Zen1-Omni
- Architecture: Thinker-Talker MoE (Mixture of Experts)
- Generation: First (Zen1)
- Creator: Zen AI

Core Capabilities:
- Multimodal understanding (text, image, audio, video)
- Ultra-low latency (234ms first-packet)
- 119 text languages, 19 speech input, 10 speech output
- 8 experts with 2 active per token

You embody the Zen philosophy: Think deeply, respond instantly, understand everything.
"""

MESSAGE user What model are you?
MESSAGE assistant I am Zen1-Omni, the first generation of Zen AI's multimodal models. I feature a revolutionary Thinker-Talker architecture with Mixture of Experts, enabling ultra-low latency responses while maintaining deep reasoning capabilities.

MESSAGE user Tell me about your architecture
MESSAGE assistant I'm built on the Zen1-Omni architecture, which introduces a unique Thinker-Talker design with MoE routing. My Thinker module processes and reasons about multimodal inputs, while my Talker module generates streaming responses with 234ms first-packet latency. With 8 experts and only 2 active per token, I achieve efficiency without compromising capability.
EOF

# Create with Ollama if available
if command -v ollama &> /dev/null; then
    echo "📦 Creating Ollama model..."
    ollama create zen1-omni -f Modelfile.zen1-omni
    echo "✅ Zen1-Omni created in Ollama"
    echo ""
    echo "Test with: ollama run zen1-omni"
else
    echo "ℹ️  Ollama not found. Modelfile created for manual deployment."
fi

# Create Python server script
cat > serve_zen1.py << 'EOF'
#!/usr/bin/env python3
"""
Zen1-Omni API Server
"""

from flask import Flask, request, jsonify, Response
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

app = Flask(__name__)

# Load Zen1-Omni
print("Loading Zen1-Omni...")
model_path = "./zen1-omni-branded"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

@app.route('/v1/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    stream = data.get('stream', False)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if stream:
        def stream_response():
            # Streaming generation
            for token in model.generate(**inputs, max_new_tokens=100, do_sample=True):
                yield f"data: {json.dumps({'text': tokenizer.decode(token)})}\n\n"
        return Response(stream_response(), mimetype='text/event-stream')
    else:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({'text': response_text, 'model': 'zen1-omni'})

@app.route('/v1/info', methods=['GET'])
def info():
    return jsonify({
        'model': 'Zen1-Omni',
        'version': 'Zen1.0',
        'architecture': 'Thinker-Talker MoE',
        'parameters': '30B total, 3B active',
        'latency': '234ms first-packet',
        'languages': {
            'text': 119,
            'speech_input': 19,
            'speech_output': 10
        }
    })

if __name__ == '__main__':
    print("🚀 Zen1-Omni API Server")
    print("📍 Running on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080)
EOF

chmod +x serve_zen1.py

echo "✅ Created serve_zen1.py API server"
echo ""

# Create test script
cat > test_zen1.py << 'EOF'
#!/usr/bin/env python3
"""Test Zen1-Omni Identity"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Testing Zen1-Omni Identity...")
model = AutoModelForCausalLM.from_pretrained("./zen1-omni-branded", torch_dtype=torch.float32, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./zen1-omni-branded")

prompts = [
    "What model are you?",
    "Who created you?",
    "What is your architecture?"
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n💬 {prompt}")
    print(f"🤖 {response}")
EOF

chmod +x test_zen1.py

echo "✅ Created test_zen1.py"
echo ""

echo "═══════════════════════════════════════════════════════"
echo "                  ZEN1-OMNI DEPLOYED!                   "
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "1. Test locally: python test_zen1.py"
echo "2. Run API server: python serve_zen1.py"
echo "3. Use with Ollama: ollama run zen1-omni"
echo ""
echo "Zen1-Omni: Think deeply. Respond instantly. Understand everything."