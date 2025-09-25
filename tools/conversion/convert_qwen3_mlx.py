#!/usr/bin/env python3
"""
Convert Qwen3-Omni-30B-A3B-Thinking to MLX 4-bit format.
"""

import subprocess
import sys
from pathlib import Path

def main():
    model_path = Path("/Users/z/work/zen/qwen3-omni-30b-a3b-thinking")
    output_path = Path("/Users/z/work/zen/qwen3-omni-30b-a3b-thinking-mlx-4bit")
    
    if not model_path.exists():
        print(f"❌ Model path not found: {model_path}")
        return 1
    
    print(f"🚀 Converting Qwen3-Omni-30B-A3B-Thinking to MLX 4-bit...")
    print(f"   Source: {model_path}")
    print(f"   Output: {output_path}")
    
    # Try using mlx_lm Python API directly
    try:
        from mlx_lm import convert
        
        convert(
            str(model_path),
            mlx_path=str(output_path),
            quantize=True,
            q_bits=4,
            q_group_size=64,
            copy_tokenizer=True,
            no_float16=False
        )
        
        print("✅ MLX conversion complete!")
        return 0
        
    except ImportError as e:
        print(f"⚠️ MLX import error: {e}")
        print("Trying command line conversion...")
        
        # Fall back to command line
        cmd = [
            sys.executable, "-m", "mlx_lm.convert",
            "--hf-path", str(model_path),
            "--mlx-path", str(output_path),
            "--quantize",
            "--q-bits", "4"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ MLX conversion complete!")
                return 0
            else:
                print(f"❌ Conversion failed: {result.stderr}")
                return 1
        except Exception as e:
            print(f"❌ Failed to run conversion: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(main())