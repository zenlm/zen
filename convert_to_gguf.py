#!/usr/bin/env python3
"""
Convert Hanzo Zen-1 model to GGUF format for Ollama
"""

import os
import sys
from pathlib import Path
import subprocess
import shutil

print("""
╔═══════════════════════════════════════════════════════╗
║         CONVERT HANZO ZEN-1 TO GGUF FORMAT           ║
║               For Ollama and llama.cpp                ║
╚═══════════════════════════════════════════════════════╝
""")

model_path = Path("gym-output/model")
if not model_path.exists():
    print("❌ Model not found at gym-output/model/")
    print("   Run gym.py first to train the model")
    sys.exit(1)

print("✅ Model found at gym-output/model/")

# Check for llama-quantize
llama_quantize = Path("llama.cpp/build/bin/llama-quantize")
if not llama_quantize.exists():
    print("⚠️  llama-quantize not found, building llama.cpp...")
    os.system("cd llama.cpp && cmake -B build && cmake --build build --config Release -j 8")

# Alternative: Use Ollama to create GGUF
print("\n📦 Creating Ollama-compatible model...")

# Create Modelfile
modelfile_content = f"""# Hanzo Zen-1 - Fine-tuned on Apple Silicon
FROM {model_path.absolute()}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1
PARAMETER stop <|im_end|>

SYSTEM You are Hanzo Zen-1, fine-tuned with knowledge of the Hanzo AI ecosystem including @hanzo/ui components, MCP tools, LLM Gateway, and all Hanzo SDKs.

TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>\"\"\"
"""

modelfile_path = Path("Modelfile.hanzo-zen1")
modelfile_path.write_text(modelfile_content)
print(f"📝 Created Modelfile: {modelfile_path}")

# Check if Ollama is installed
ollama_installed = subprocess.run(["which", "ollama"], capture_output=True).returncode == 0

if ollama_installed:
    print("\n🚀 Creating Ollama model...")
    print("\nRun these commands:")
    print(f"  ollama create hanzo-zen1 -f {modelfile_path}")
    print("  ollama run hanzo-zen1")
else:
    print("\n⚠️  Ollama not installed")
    print("Install with: curl -fsSL https://ollama.com/install.sh | sh")

# Alternative approach using Python
print("\n🔧 Alternative: Direct GGUF conversion")
print("\n1. Install llama-cpp-python:")
print("   pip install llama-cpp-python")

print("\n2. Use this Python script:")
conversion_script = '''
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained("gym-output/model")
tokenizer = AutoTokenizer.from_pretrained("gym-output/model")

# Save in format llama.cpp can use
model.save_pretrained("model-for-gguf", safe_serialization=False)
tokenizer.save_pretrained("model-for-gguf")

print("Model ready for GGUF conversion at model-for-gguf/")
'''

script_path = Path("prepare_for_gguf.py")
script_path.write_text(conversion_script)
print(f"   Saved script: {script_path}")

print("\n3. Then convert:")
print("   python llama.cpp/convert.py model-for-gguf --outtype q4_0")

# Check model size
model_size = sum(f.stat().st_size for f in model_path.glob("*")) / (1024**3)
print(f"\n📊 Model Statistics:")
print(f"  Original size: {model_size:.2f} GB")
print(f"  Q4_0 estimated: ~{model_size * 0.25:.2f} GB")
print(f"  Q8_0 estimated: ~{model_size * 0.5:.2f} GB")

print("\n✅ Conversion setup complete!")
print("\n🎯 Quick test with transformers:")
print("   from transformers import pipeline")
print(f"   pipe = pipeline('text-generation', '{model_path}')")
print("   print(pipe('What is @hanzo/ui?', max_length=50))")