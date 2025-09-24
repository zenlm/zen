#!/usr/bin/env python3
"""
Monitor Qwen3-Omni-30B-A3B-Thinking download and auto-convert to MLX and GGUF.
"""

import os
import time
import subprocess
from pathlib import Path
import json

class ModelPipeline:
    def __init__(self):
        self.model_dir = Path("/Users/z/work/zen/qwen3-omni-30b-a3b-thinking")
        self.mlx_output = Path("/Users/z/work/zen/qwen3-omni-30b-a3b-thinking-mlx-4bit")
        self.gguf_output = Path("/Users/z/work/zen/qwen3-omni-30b-a3b-thinking-gguf")
        
    def check_download_complete(self):
        """Check if all model files are downloaded."""
        if not self.model_dir.exists():
            return False
            
        # Check for index file
        index_file = self.model_dir / "model.safetensors.index.json"
        if not index_file.exists():
            return False
            
        # Parse index to get required files
        with open(index_file) as f:
            index = json.load(f)
            
        required_files = set(index['weight_map'].values())
        
        # Check all files exist
        for file_name in required_files:
            if not (self.model_dir / file_name).exists():
                return False
                
        return True
    
    def convert_to_mlx(self):
        """Convert model to MLX 4-bit format."""
        print("\n🚀 Starting MLX conversion...")
        
        cmd = [
            "mlx_lm.convert",
            "--hf-path", str(self.model_dir),
            "--mlx-path", str(self.mlx_output),
            "--quantize",
            "--q-bits", "4"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ MLX conversion complete!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ MLX conversion failed: {e}")
            return False
    
    def test_mlx_model(self):
        """Quick test of MLX model."""
        print("\n🧪 Testing MLX model...")
        
        test_script = '''
from mlx_lm import load, generate

model_path = "{}"
model, tokenizer = load(model_path)
            
prompt = "Hello, I am"
response = generate(model, tokenizer, prompt, max_tokens=50)
print("Test response:", response)
'''.format(self.mlx_output)
        
        try:
            subprocess.run(["python", "-c", test_script], check=True)
            print("✅ MLX model test passed!")
            return True
        except subprocess.CalledProcessError:
            print("⚠️ MLX test failed - model may need debugging")
            return False
    
    def prepare_gguf_conversion(self):
        """Prepare GGUF conversion script."""
        print("\n📦 Preparing GGUF conversion...")
        
        script = '''#!/bin/bash
# GGUF Conversion Script for Qwen3-Omni-30B-A3B-Thinking

MODEL_DIR="{}"
OUTPUT_DIR="{}"

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
'''.format(self.model_dir, self.gguf_output)
        
        script_path = Path("convert_to_gguf.sh")
        script_path.write_text(script)
        script_path.chmod(0o755)
        print(f"✅ GGUF conversion script created: {script_path}")
        
    def monitor_and_convert(self):
        """Main monitoring loop."""
        print("🔍 Monitoring download progress...")
        print(f"   Model directory: {self.model_dir}")
        
        while True:
            if self.check_download_complete():
                print("\n✅ Download complete! Starting conversions...")
                
                # Convert to MLX
                if self.convert_to_mlx():
                    self.test_mlx_model()
                
                # Prepare GGUF
                self.prepare_gguf_conversion()
                
                print("\n🎉 Pipeline complete!")
                break
            else:
                # Check how many files exist
                if self.model_dir.exists():
                    safetensor_files = list(self.model_dir.glob("*.safetensors"))
                    print(f"📊 Progress: {len(safetensor_files)}/16 model files downloaded")
                else:
                    print("⏳ Waiting for download to start...")
                    
                time.sleep(60)  # Check every minute

def main():
    pipeline = ModelPipeline()
    pipeline.monitor_and_convert()

if __name__ == "__main__":
    main()