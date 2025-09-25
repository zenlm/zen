#!/usr/bin/env python3
"""
Supra Nexus O1 Quantization Tool
Creates 4-bit quantized versions for edge deployment
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

# Configuration
BASE_DIR = Path("/Users/z/work/supra/o1/models")
OUTPUT_DIR = Path("/Users/z/work/supra/o1/models/quantized")

MODELS = {
    "thinking": BASE_DIR / "supra-nexus-o1-thinking-fused",
    "instruct": BASE_DIR / "supra-nexus-o1-instruct-fused"
}

def run_command(cmd: list, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Execute command with proper error handling."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def check_mlx() -> bool:
    """Check if MLX is installed."""
    try:
        import mlx
        return True
    except ImportError:
        return False

def quantize_mlx(model_path: Path, output_path: Path, bits: int = 4) -> None:
    """Quantize model using MLX."""
    print(f"Quantizing with MLX to {bits}-bit...")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # MLX quantization command
    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", str(model_path),
        "--mlx-path", str(output_path),
        "--quantize",
        "--q-bits", str(bits)
    ]
    
    result = run_command(cmd)
    print(f"✓ MLX quantization complete: {output_path}")

def create_gguf(model_path: Path, output_path: Path, quant_type: str = "Q4_K_M") -> None:
    """Create GGUF format for llama.cpp."""
    print(f"Creating GGUF format ({quant_type})...")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check for llama.cpp convert script
    convert_script = Path.home() / "work" / "zen" / "llama.cpp" / "convert_hf_to_gguf.py"
    quantize_bin = Path.home() / "work" / "zen" / "llama.cpp" / "llama-quantize"
    
    if not convert_script.exists():
        print(f"Warning: llama.cpp not found at {convert_script.parent}")
        print("Skipping GGUF conversion")
        return
    
    # Convert to GGUF F16
    base_gguf = output_path / f"{model_path.name}-F16.gguf"
    cmd = [
        sys.executable, str(convert_script),
        str(model_path),
        "--outtype", "f16",
        "--outfile", str(base_gguf)
    ]
    
    print("Converting to GGUF F16...")
    run_command(cmd)
    
    # Quantize to requested format
    if quantize_bin.exists():
        quant_gguf = output_path / f"{model_path.name}-{quant_type}.gguf"
        cmd = [
            str(quantize_bin),
            str(base_gguf),
            str(quant_gguf),
            quant_type
        ]
        
        print(f"Quantizing to {quant_type}...")
        run_command(cmd)
        print(f"✓ GGUF quantization complete: {quant_gguf}")
        
        # Clean up F16 if quantization succeeded
        base_gguf.unlink()
    else:
        print(f"✓ GGUF F16 created: {base_gguf}")

def update_config_for_quantization(model_path: Path, output_path: Path, bits: int) -> None:
    """Update model config for quantized version."""
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        # Add quantization metadata
        config["quantization_config"] = {
            "bits": bits,
            "quant_method": "mlx" if check_mlx() else "bitsandbytes",
            "version": "1.0"
        }
        
        output_config = output_path / "config.json"
        with open(output_config, "w") as f:
            json.dump(config, f, indent=2)
        
        # Copy other necessary files
        for file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            src = model_path / file
            if src.exists():
                shutil.copy2(src, output_path / file)

def main():
    """Main quantization workflow."""
    print("Supra Nexus O1 Quantization Tool")
    print("=" * 60)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for MLX
    has_mlx = check_mlx()
    if has_mlx:
        print("✓ MLX available for Apple Silicon optimization")
    else:
        print("ℹ MLX not found - using standard quantization")
        print("  Install: pip install mlx mlx-lm")
    
    # Process each model
    for variant, model_path in MODELS.items():
        if not model_path.exists():
            print(f"Warning: {variant} model not found at {model_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: supra-nexus-o1-{variant}")
        print(f"{'='*60}")
        
        # MLX quantization (4-bit)
        if has_mlx:
            mlx_output = OUTPUT_DIR / f"supra-nexus-o1-{variant}-4bit-mlx"
            try:
                quantize_mlx(model_path, mlx_output, bits=4)
                update_config_for_quantization(model_path, mlx_output, bits=4)
            except Exception as e:
                print(f"MLX quantization failed: {e}")
        
        # GGUF quantization
        gguf_output = OUTPUT_DIR / f"supra-nexus-o1-{variant}-gguf"
        try:
            create_gguf(model_path, gguf_output, quant_type="Q4_K_M")
        except Exception as e:
            print(f"GGUF creation failed: {e}")
    
    print("\n" + "=" * 60)
    print("Quantization Complete!")
    print("=" * 60)
    print(f"\nQuantized models saved to: {OUTPUT_DIR}")
    
    # List output files
    if OUTPUT_DIR.exists():
        print("\nGenerated files:")
        for file in sorted(OUTPUT_DIR.rglob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  {file.relative_to(OUTPUT_DIR)} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()