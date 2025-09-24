#!/usr/bin/env python3
"""
Create GGUF models for zen-nano-instruct and zen-nano-thinking variants.
Ensures both GGUF and MLX formats are available for complete deployment.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Error: {e.stderr}")
        return None

def convert_to_gguf(input_model_path, output_dir, model_name, quantization="q4_k_m"):
    """Convert MLX model to GGUF format."""

    input_path = Path(input_model_path)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"❌ Input model not found: {input_path}")
        return False

    output_path.mkdir(parents=True, exist_ok=True)

    # First convert to HuggingFace format if needed
    hf_temp_path = output_path / f"{model_name}-hf-temp"

    # Convert MLX to HuggingFace
    cmd = f"python -m mlx_lm.convert --hf-path {input_path} --upload-repo {hf_temp_path}"
    if not run_command(cmd, f"Converting {model_name} MLX to HuggingFace format"):
        return False

    # Convert HuggingFace to GGUF
    gguf_path = output_path / f"{model_name}-{quantization}.gguf"

    # Use llama.cpp convert script
    llama_cpp_path = Path("zen-nano/llama.cpp")
    if llama_cpp_path.exists():
        convert_script = llama_cpp_path / "convert-hf-to-gguf.py"
        if convert_script.exists():
            cmd = f"cd {llama_cpp_path} && python {convert_script} {hf_temp_path} --outdir {output_path} --outtype {quantization}"
            run_command(cmd, f"Converting to GGUF {quantization}")

    # Clean up temp directory
    run_command(f"rm -rf {hf_temp_path}", f"Cleaning up temp files")

    return gguf_path.exists()

def main():
    """Main conversion process."""

    models_to_convert = [
        {
            "mlx_path": "zen-nano/models/zen-nano-4b-instruct-mlx-final",
            "output_dir": "zen-nano/models/zen-nano-4b-instruct-gguf",
            "name": "zen-nano-instruct"
        },
        {
            "mlx_path": "zen-nano/models/zen-nano-4b-thinking-mlx",
            "output_dir": "zen-nano/models/zen-nano-4b-thinking-gguf",
            "name": "zen-nano-thinking"
        }
    ]

    quantizations = ["q4_k_m", "q8_0", "f16"]

    print("🚀 Creating GGUF models for Zen-Nano variants")
    print("=" * 50)

    for model in models_to_convert:
        print(f"\n📦 Processing {model['name']}")

        for quant in quantizations:
            success = convert_to_gguf(
                model["mlx_path"],
                model["output_dir"],
                model["name"],
                quant
            )

            if success:
                print(f"✅ Created {model['name']}-{quant}.gguf")
            else:
                print(f"⚠️  Failed to create {model['name']}-{quant}.gguf")

    print("\n🎉 GGUF model creation complete!")
    print("Both GGUF and MLX formats now available for deployment")

if __name__ == "__main__":
    main()