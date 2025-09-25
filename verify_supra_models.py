#!/usr/bin/env python3
"""
Verify Supra Nexus O1 models before deployment
Minimal verification script
"""

import json
import sys
from pathlib import Path

def verify_model(model_path: Path) -> bool:
    """Verify model structure and files."""
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    print(f"\nVerifying: {model_path.name}")
    print("-" * 40)
    
    # Check path exists
    if not model_path.exists():
        print(f"✗ Model directory not found")
        return False
    
    # Check required files
    for file in required_files:
        file_path = model_path / file
        if not file_path.exists():
            print(f"✗ Missing: {file}")
            return False
        print(f"✓ Found: {file}")
    
    # Check for model weights
    safetensors = list(model_path.glob("*.safetensors"))
    if not safetensors:
        print(f"✗ No model weights (.safetensors) found")
        return False
    
    total_size = sum(f.stat().st_size for f in safetensors) / (1024**3)
    print(f"✓ Model weights: {len(safetensors)} files, {total_size:.1f} GB")
    
    # Verify config
    try:
        with open(model_path / "config.json") as f:
            config = json.load(f)
        print(f"✓ Model type: {config.get('model_type', 'unknown')}")
        print(f"✓ Hidden size: {config.get('hidden_size', 'unknown')}")
    except Exception as e:
        print(f"✗ Invalid config: {e}")
        return False
    
    return True

def main():
    """Main verification."""
    models = {
        "thinking": Path("/Users/z/work/supra/o1/models/supra-nexus-o1-thinking-fused"),
        "instruct": Path("/Users/z/work/supra/o1/models/supra-nexus-o1-instruct-fused")
    }
    
    print("Supra Nexus O1 Model Verification")
    print("=" * 40)
    
    all_valid = True
    for name, path in models.items():
        if not verify_model(path):
            all_valid = False
    
    print("\n" + "=" * 40)
    if all_valid:
        print("✓ All models verified successfully")
        print("✓ Ready for deployment")
    else:
        print("✗ Verification failed")
        print("✗ Fix issues before deployment")
        sys.exit(1)

if __name__ == "__main__":
    main()