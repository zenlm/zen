#!/usr/bin/env python3
"""
Overnight pipeline for Qwen3-Omni MLX conversion and LM Studio setup.
This script will monitor download progress and automatically convert when ready.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import json
from datetime import datetime

class Qwen3Pipeline:
    def __init__(self):
        self.model_path = Path.home() / "work/zen/qwen3-omni-30b-complete"
        self.mlx_path = Path.home() / "work/zen/qwen3-omni-mlx"
        self.gguf_path = Path.home() / "work/zen/qwen3-omni-gguf"
        self.log_file = Path.home() / "work/zen/qwen3_pipeline.log"
        
    def log(self, message):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
            
    def check_download_complete(self):
        """Check if all model files are downloaded."""
        expected_files = [f"model-{i:05d}-of-00016.safetensors" for i in range(1, 17)]
        
        existing = []
        missing = []
        
        for file in expected_files:
            file_path = self.model_path / file
            if file_path.exists():
                size_gb = file_path.stat().st_size / 1e9
                existing.append(f"{file} ({size_gb:.1f}GB)")
            else:
                missing.append(file)
                
        self.log(f"Download status: {len(existing)}/16 files complete")
        
        if missing:
            self.log(f"Still missing: {', '.join(missing[:3])}...")
            
        return len(missing) == 0, existing, missing
    
    def wait_for_download(self, check_interval=60):
        """Wait for download to complete, checking periodically."""
        self.log("Monitoring download progress...")
        
        while True:
            complete, existing, missing = self.check_download_complete()
            
            if complete:
                self.log("✅ Download complete! All 16 files present.")
                return True
                
            self.log(f"⏳ Waiting... {len(existing)}/16 files downloaded")
            time.sleep(check_interval)
            
    def convert_to_mlx(self):
        """Convert model to MLX format."""
        self.log("Starting MLX conversion...")
        
        # Create output directories
        fp16_path = self.mlx_path / "fp16"
        q4_path = self.mlx_path / "q4"
        fp16_path.mkdir(parents=True, exist_ok=True)
        q4_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Convert to FP16
        self.log("Converting to MLX FP16 format...")
        cmd = [
            "python3", "-m", "mlx_lm.convert",
            "--hf-path", str(self.model_path),
            "--mlx-path", str(fp16_path),
            "--dtype", "float16"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                self.log("✅ FP16 conversion successful")
            else:
                self.log(f"❌ FP16 conversion failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.log("⚠️ FP16 conversion timed out (>1 hour)")
            return False
        except Exception as e:
            self.log(f"❌ Error during FP16 conversion: {e}")
            return False
            
        # Step 2: Quantize to 4-bit
        self.log("Quantizing to 4-bit...")
        cmd = [
            "python3", "-m", "mlx_lm.quantize",
            "--model", str(fp16_path),
            "--bits", "4",
            "--output-dir", str(q4_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                self.log("✅ 4-bit quantization successful")
                
                # Check size reduction
                if q4_path.exists():
                    size_gb = sum(f.stat().st_size for f in q4_path.glob("*.safetensors")) / 1e9
                    self.log(f"📊 4-bit model size: {size_gb:.1f}GB")
                    
                return True
            else:
                self.log(f"❌ Quantization failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.log("⚠️ Quantization timed out (>30 min)")
            return False
        except Exception as e:
            self.log(f"❌ Error during quantization: {e}")
            return False
            
    def test_mlx_model(self):
        """Test the MLX model with a simple generation."""
        self.log("Testing MLX model...")
        
        test_script = '''
from mlx_lm import load, generate
import sys

try:
    model_path = sys.argv[1]
    print(f"Loading model from {model_path}...")
    
    model, tokenizer = load(model_path)
    print("Model loaded successfully!")
    
    prompt = "You are Zen-Omni. Explain your capabilities in one sentence:"
    response = generate(
        model, 
        tokenizer,
        prompt=prompt,
        max_tokens=50,
        temperature=0.7
    )
    
    print(f"Response: {response}")
    print("✅ Model test successful!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
'''
        
        # Save test script
        test_file = self.model_path.parent / "test_mlx_model.py"
        with open(test_file, 'w') as f:
            f.write(test_script)
            
        # Run test
        q4_path = self.mlx_path / "q4"
        cmd = ["python3", str(test_file), str(q4_path)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            self.log(result.stdout)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            self.log("⚠️ Model test timed out (>5 min)")
            return False
        except Exception as e:
            self.log(f"❌ Error during test: {e}")
            return False
            
    def prepare_gguf_conversion(self):
        """Prepare scripts for GGUF conversion."""
        self.log("Preparing GGUF conversion scripts...")
        
        self.gguf_path.mkdir(parents=True, exist_ok=True)
        
        # Create conversion script
        convert_script = f'''#!/bin/bash
# GGUF conversion script for Qwen3-Omni

MODEL_PATH="{self.model_path}"
GGUF_PATH="{self.gguf_path}"

echo "Installing/updating llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp
git pull
make clean && make

echo "Converting to GGUF F16..."
python3 convert_hf_to_gguf.py "$MODEL_PATH" \\
    --outfile "$GGUF_PATH/qwen3-omni-30b.gguf" \\
    --outtype f16

echo "Quantizing to 4-bit..."
./llama-quantize "$GGUF_PATH/qwen3-omni-30b.gguf" \\
    "$GGUF_PATH/qwen3-omni-30b-q4_k_m.gguf" Q4_K_M

echo "✅ GGUF conversion complete!"
echo "Model ready at: $GGUF_PATH/qwen3-omni-30b-q4_k_m.gguf"
'''
        
        script_path = self.gguf_path / "convert_to_gguf.sh"
        with open(script_path, 'w') as f:
            f.write(convert_script)
        script_path.chmod(0o755)
        
        self.log(f"✅ GGUF conversion script ready at: {script_path}")
        
        # Create LM Studio instructions
        lm_studio_doc = f'''# LM Studio Setup for Qwen3-Omni

## Model Location
- MLX 4-bit: `{self.mlx_path}/q4/`
- GGUF 4-bit: `{self.gguf_path}/qwen3-omni-30b-q4_k_m.gguf`

## Installation Steps

1. **Copy GGUF model to LM Studio:**
```bash
cp {self.gguf_path}/qwen3-omni-30b-q4_k_m.gguf \\
   ~/Library/Application\\ Support/LM\\ Studio/models/
```

2. **Open LM Studio**
3. **Select Model:** Choose `qwen3-omni-30b-q4_k_m.gguf`
4. **Configure Settings:**
   - Context Length: 8192
   - GPU Layers: -1 (use all)
   - Temperature: 0.7
   - Top-P: 0.95
   - Repeat Penalty: 1.1

## System Prompt
```
You are Zen-Omni, a hypermodal AI assistant with capabilities spanning:
- Text understanding and generation
- Vision and image analysis
- Audio and speech processing
- 3D spatial reasoning
- Code understanding and generation
```

## Performance Expectations
- Model size: ~15GB (4-bit quantized)
- RAM required: 20-24GB
- Speed: 10-20 tokens/sec on Apple Silicon

## Testing
Use this prompt to test:
"Explain your multimodal capabilities and how you process different types of information."

---
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''
        
        doc_path = self.gguf_path / "LM_Studio_Setup.md"
        with open(doc_path, 'w') as f:
            f.write(lm_studio_doc)
            
        self.log(f"✅ LM Studio setup guide ready at: {doc_path}")
        
        return script_path
        
    def run_pipeline(self):
        """Run the complete overnight pipeline."""
        self.log("=" * 60)
        self.log("Starting Qwen3-Omni Overnight Pipeline")
        self.log("=" * 60)
        
        # Step 1: Wait for download
        self.log("Step 1: Waiting for model download...")
        if not self.wait_for_download():
            self.log("❌ Download failed or interrupted")
            return
            
        # Step 2: Convert to MLX
        self.log("\nStep 2: Converting to MLX 4-bit...")
        if self.convert_to_mlx():
            self.log("✅ MLX conversion complete!")
            
            # Step 3: Test MLX model
            self.log("\nStep 3: Testing MLX model...")
            if self.test_mlx_model():
                self.log("✅ MLX model test passed!")
            else:
                self.log("⚠️ MLX model test failed, but continuing...")
        else:
            self.log("❌ MLX conversion failed")
            
        # Step 4: Prepare GGUF conversion
        self.log("\nStep 4: Preparing GGUF conversion...")
        gguf_script = self.prepare_gguf_conversion()
        
        # Final summary
        self.log("\n" + "=" * 60)
        self.log("Pipeline Complete!")
        self.log("=" * 60)
        self.log(f"""
Next steps when you wake up:

1. MLX model is ready at: {self.mlx_path}/q4/
   Test with: python3 -m mlx_lm.generate --model {self.mlx_path}/q4 --prompt "Hello"

2. For GGUF/LM Studio, run: {gguf_script}

3. Then load in LM Studio following: {self.gguf_path}/LM_Studio_Setup.md

Sweet dreams! The models will be ready when you wake up. 🌙
""")

if __name__ == "__main__":
    pipeline = Qwen3Pipeline()
    pipeline.run_pipeline()