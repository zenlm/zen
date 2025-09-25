#!/usr/bin/env python3
"""
Download Qwen3 base models for Zen family.
Minimal dependencies - uses only huggingface_hub.
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download, hf_hub_download

# Model configurations
MODELS = {
    "zen-nano-instruct": {
        "repo_id": "Qwen/Qwen3-4B-Instruct-2507",
        "local_dir": "Qwen3-4B-Instruct-2507",
        "description": "4B Instruct model for zen-nano-instruct"
    },
    "zen-nano-thinking": {
        "repo_id": "Qwen/Qwen3-4B-Thinking-2507",
        "local_dir": "Qwen3-4B-Thinking-2507",
        "description": "4B Thinking model for zen-nano-thinking"
    },
    "zen-omni": {
        "repo_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "local_dir": "Qwen3-Omni-30B-A3B-Instruct",
        "description": "30B Omni Instruct model for zen-omni"
    },
    "zen-omni-thinking": {
        "repo_id": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "local_dir": "Qwen3-Omni-30B-A3B-Thinking",
        "description": "30B Omni Thinking model for zen-omni-thinking"
    },
    "zen-omni-captioner": {
        "repo_id": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
        "local_dir": "Qwen3-Omni-30B-A3B-Captioner",
        "description": "30B Omni Captioner model for zen-omni-captioner"
    }
}

BASE_DIR = Path("/Users/z/work/zen/base-models")


def calculate_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Calculate hash of a file."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_model(zen_variant: str, force: bool = False) -> bool:
    """
    Download a single model.
    
    Args:
        zen_variant: Key from MODELS dict
        force: Force re-download even if exists
    
    Returns:
        True if successful, False otherwise
    """
    if zen_variant not in MODELS:
        print(f"❌ Unknown model variant: {zen_variant}")
        return False
    
    config = MODELS[zen_variant]
    local_path = BASE_DIR / config["local_dir"]
    
    print(f"\n{'='*60}")
    print(f"📦 Model: {zen_variant}")
    print(f"📝 Description: {config['description']}")
    print(f"🔗 Repo: {config['repo_id']}")
    print(f"📂 Local: {local_path}")
    print(f"{'='*60}")
    
    # Check if already exists
    if local_path.exists() and not force:
        model_files = list(local_path.glob("*.safetensors")) + \
                     list(local_path.glob("*.bin")) + \
                     list(local_path.glob("*.pt"))
        if model_files:
            print(f"✅ Model already exists at {local_path}")
            print(f"   Found {len(model_files)} model files")
            print(f"   Use --force to re-download")
            return True
    
    # Create directory
    local_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"⬇️  Downloading {config['repo_id']}...")
        print(f"   This may take a while for large models...")
        
        # Download the model
        snapshot_download(
            repo_id=config["repo_id"],
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4  # Parallel downloads
        )
        
        print(f"✅ Successfully downloaded to {local_path}")
        
        # List downloaded files
        print(f"\n📁 Downloaded files:")
        for ext in ["*.json", "*.txt", "*.safetensors", "*.bin", "*.pt", "*.model"]:
            files = list(local_path.glob(ext))
            if files:
                print(f"   {ext}: {len(files)} file(s)")
                for f in files[:3]:  # Show first 3 files
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"      - {f.name} ({size_mb:.1f} MB)")
                if len(files) > 3:
                    print(f"      ... and {len(files) - 3} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {config['repo_id']}: {e}")
        return False


def verify_model(zen_variant: str) -> bool:
    """Verify a downloaded model."""
    if zen_variant not in MODELS:
        print(f"❌ Unknown model variant: {zen_variant}")
        return False
    
    config = MODELS[zen_variant]
    local_path = BASE_DIR / config["local_dir"]
    
    if not local_path.exists():
        print(f"❌ Model not found at {local_path}")
        return False
    
    print(f"\n🔍 Verifying {zen_variant}...")
    
    # Check for essential files
    essential_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing = []
    for fname in essential_files:
        if not (local_path / fname).exists():
            missing.append(fname)
    
    if missing:
        print(f"⚠️  Missing essential files: {', '.join(missing)}")
    
    # Check for model files
    model_files = list(local_path.glob("*.safetensors")) + \
                 list(local_path.glob("*.bin")) + \
                 list(local_path.glob("*.pt"))
    
    if not model_files:
        print(f"❌ No model files found (.safetensors, .bin, or .pt)")
        return False
    
    print(f"✅ Found {len(model_files)} model file(s)")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
    print(f"📊 Total size: {total_size / (1024**3):.2f} GB")
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Qwen3 models for Zen family")
    parser.add_argument(
        "models",
        nargs="*",
        choices=list(MODELS.keys()) + ["all"],
        help="Models to download (default: all)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list:
        print("\n📋 Available models for Zen family:")
        for key, config in MODELS.items():
            print(f"\n  {key}:")
            print(f"    Description: {config['description']}")
            print(f"    Repo: {config['repo_id']}")
            print(f"    Local dir: {config['local_dir']}")
        return 0
    
    # Ensure base directory exists
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 Base directory: {BASE_DIR}")
    
    # Determine which models to process
    if not args.models or "all" in args.models:
        models_to_process = list(MODELS.keys())
    else:
        models_to_process = args.models
    
    # Verify models
    if args.verify:
        print("\n🔍 Verifying models...")
        for variant in models_to_process:
            verify_model(variant)
        return 0
    
    # Download models
    print(f"\n🚀 Downloading {len(models_to_process)} model(s)...")
    results = []
    
    for variant in models_to_process:
        success = download_model(variant, force=args.force)
        results.append((variant, success))
        
        # Verify after download
        if success:
            verify_model(variant)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Download Summary:")
    print(f"{'='*60}")
    
    successful = [v for v, s in results if s]
    failed = [v for v, s in results if not s]
    
    if successful:
        print(f"✅ Successful ({len(successful)}):")
        for variant in successful:
            print(f"   - {variant}")
    
    if failed:
        print(f"❌ Failed ({len(failed)}):")
        for variant in failed:
            print(f"   - {variant}")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())