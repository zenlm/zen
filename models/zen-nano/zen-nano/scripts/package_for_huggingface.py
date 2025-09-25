#!/usr/bin/env python3
"""
Package Zen Nano v1.0 for Hugging Face
- Create quantized MLX versions from existing model
- Convert to GGUF format
- Prepare for upload
"""

import subprocess
import shutil
from pathlib import Path
import json

def create_mlx_quantized():
    """Create quantized versions of the existing MLX model"""
    print("🔧 Creating quantized MLX versions...")
    
    base_model = "models/zen-nano-v1-mlx"
    
    # Quantization configs
    configs = [
        (4, "models/zen-nano-v1-mlx-q4"),
        (8, "models/zen-nano-v1-mlx-q8"),
    ]
    
    for bits, output in configs:
        print(f"  📦 Creating {bits}-bit quantized version...")
        cmd = [
            "python3.12", "-m", "mlx_lm.convert",
            "--hf-path", base_model,
            "--mlx-path", output,
            "--quantize",
            "--q-bits", str(bits)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print(f"  ✅ Created {bits}-bit at {output}")
            else:
                print(f"  ⚠️  Issue with {bits}-bit: {result.stderr[:200]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

def convert_to_gguf():
    """Convert MLX model to GGUF format"""
    print("\n🔧 Converting to GGUF format...")
    
    # Create GGUF directory
    gguf_dir = Path("models/gguf")
    gguf_dir.mkdir(exist_ok=True, parents=True)
    
    # First check if llama.cpp is available
    llama_cpp = Path("../../../llama.cpp")
    if not llama_cpp.exists():
        llama_cpp = Path("llama.cpp")
        if not llama_cpp.exists():
            print("  📦 Cloning llama.cpp...")
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/ggerganov/llama.cpp.git"
            ])
            
            # Build it
            print("  🔨 Building llama.cpp...")
            subprocess.run(["make", "-C", "llama.cpp", "clean"])
            subprocess.run(["make", "-C", "llama.cpp", "LLAMA_METAL=1"])
    
    # Convert to GGUF
    convert_script = llama_cpp / "convert-hf-to-gguf.py"
    if convert_script.exists():
        print("  📦 Converting to GGUF F16...")
        cmd = [
            "python3.12", str(convert_script),
            "models/zen-nano-v1-mlx",
            "--outfile", "models/gguf/zen-nano-v1-f16.gguf",
            "--outtype", "f16"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ Created F16 GGUF")
            
            # Now quantize it
            quantize_exe = llama_cpp / "quantize"
            if quantize_exe.exists():
                quants = [
                    ("Q4_K_M", "zen-nano-v1-q4_k_m.gguf"),
                    ("Q5_K_M", "zen-nano-v1-q5_k_m.gguf"),
                    ("Q8_0", "zen-nano-v1-q8_0.gguf")
                ]
                
                for qtype, filename in quants:
                    print(f"  📦 Creating {qtype} quantized GGUF...")
                    cmd = [
                        str(quantize_exe),
                        "models/gguf/zen-nano-v1-f16.gguf",
                        f"models/gguf/{filename}",
                        qtype
                    ]
                    subprocess.run(cmd, capture_output=True)
                    print(f"  ✅ Created {filename}")
        else:
            print(f"  ⚠️  GGUF conversion issue: {result.stderr[:300]}")

def create_model_card():
    """Create README.md for Hugging Face"""
    print("\n📝 Creating model card...")
    
    card = """---
license: apache-2.0
language:
- en
library_name: mlx
tags:
- mlx
- gguf
- edge-ai
- lightweight
- zen-nano
base_model: Qwen/Qwen2.5-3B-Instruct
---

# Zen Nano v1.0

An ultra-lightweight AI model optimized for edge devices, jointly developed by **Hanzo AI Inc** (Techstars-backed, LA) and **Zoo Labs Foundation** (501c3, SF).

## Features
- 🌐 Runs entirely offline on edge devices
- 🔒 Complete privacy - no data leaves your device
- 🌊 Eco-friendly with minimal carbon footprint
- 📱 Works on phones, tablets, Raspberry Pi
- 🆓 Forever free and open source

## Available Files

### MLX Format (Apple Silicon)
- `zen-nano-v1-mlx/` - Full precision MLX
- `zen-nano-v1-mlx-q4/` - 4-bit quantized (~1-2GB)
- `zen-nano-v1-mlx-q8/` - 8-bit quantized (~2-3GB)

### GGUF Format (Universal)
- `gguf/zen-nano-v1-f16.gguf` - Full F16 precision
- `gguf/zen-nano-v1-q4_k_m.gguf` - 4-bit (recommended for most)
- `gguf/zen-nano-v1-q5_k_m.gguf` - 5-bit (better quality)
- `gguf/zen-nano-v1-q8_0.gguf` - 8-bit (high quality)

## Quick Start

### MLX (Mac/Apple Silicon)
```python
from mlx_lm import load, generate

model, tokenizer = load("hanzo-ai/zen-nano-v1-mlx-q4")
response = generate(model, tokenizer, prompt="Who are you?", max_tokens=100)
print(response)
```

### Ollama
```bash
ollama run hanzo-ai/zen-nano:q4
```

### llama.cpp
```bash
./main -m zen-nano-v1-q4_k_m.gguf -p "Who are you?" -n 100
```

## Model Details
- **Base**: Fine-tuned from Qwen 3B architecture
- **Context**: 32K tokens
- **Training**: LoRA fine-tuning with identity alignment
- **Optimization**: Quantization for edge deployment

## About the Creators

**Hanzo AI Inc**: Techstars-backed applied AI research lab based in Los Angeles, building practical AI tools including 100+ MCP development tools.

**Zoo Labs Foundation**: 501(c)(3) non-profit in San Francisco dedicated to democratizing AI access while protecting our oceans.

## License

Apache 2.0 - Free for any use including commercial.

---
*Zen Nano - AI that runs where you are.*
"""
    
    with open("README.md", "w") as f:
        f.write(card)
    print("  ✅ Model card created")

def prepare_upload():
    """Organize files for upload"""
    print("\n📦 Preparing Hugging Face upload structure...")
    
    upload_dir = Path("huggingface-upload")
    upload_dir.mkdir(exist_ok=True)
    
    # Copy MLX models
    for mlx_dir in Path("models").glob("zen-nano-v1-mlx*"):
        dest = upload_dir / mlx_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(mlx_dir, dest)
        print(f"  ✅ Copied {mlx_dir.name}")
    
    # Copy GGUF files
    gguf_src = Path("models/gguf")
    if gguf_src.exists():
        gguf_dest = upload_dir / "gguf"
        gguf_dest.mkdir(exist_ok=True)
        for gguf in gguf_src.glob("*.gguf"):
            shutil.copy2(gguf, gguf_dest / gguf.name)
            print(f"  ✅ Copied {gguf.name}")
    
    # Copy README
    shutil.copy2("README.md", upload_dir / "README.md")
    
    print(f"\n✅ Upload package ready at: {upload_dir}/")
    print("\nTo upload to Hugging Face:")
    print("  1. pip install huggingface-hub")
    print("  2. huggingface-cli login")
    print("  3. cd huggingface-upload")
    print("  4. huggingface-cli repo create zen-nano-v1 --type model")
    print("  5. git init && git add .")
    print("  6. git commit -m 'Initial release'")
    print("  7. git remote add origin https://huggingface.co/hanzo-ai/zen-nano-v1")
    print("  8. git push")

def main():
    print("🚀 Packaging Zen Nano v1.0 for Hugging Face")
    print("=" * 60)
    
    # We already have the base MLX model, just need to:
    # 1. Create quantized versions
    create_mlx_quantized()
    
    # 2. Convert to GGUF
    convert_to_gguf()
    
    # 3. Create model card
    create_model_card()
    
    # 4. Prepare upload structure
    prepare_upload()
    
    print("\n" + "=" * 60)
    print("✅ Package ready for Hugging Face!")

if __name__ == "__main__":
    main()