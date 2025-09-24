#!/usr/bin/env python3
"""
Test script to verify Zen1 setup
"""

import sys
from pathlib import Path

def test_setup():
    """Run all setup tests"""
    print("🧪 Testing Zen1 Setup")
    print("=" * 50)

    # Check Python version
    py_version = sys.version_info
    print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major == 3 and py_version.minor >= 9:
        print("  Python version OK")
    else:
        print("  ⚠️  Python 3.9+ recommended")

    # Check dependencies
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
        return False

    try:
        import peft
        print(f"✓ PEFT {peft.__version__}")
    except ImportError:
        print("✗ PEFT not installed")
        return False

    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
    except ImportError:
        print("✗ Datasets not installed")
        return False

    # Check data files
    print("\n📁 Training Data:")
    data_dir = Path(__file__).parent / "data"
    if data_dir.exists():
        for file in ["reasoning.jsonl", "conversations.jsonl"]:
            file_path = data_dir / file
            if file_path.exists():
                with open(file_path) as f:
                    lines = sum(1 for _ in f)
                print(f"  ✓ {file}: {lines} examples")
            else:
                print(f"  ✗ {file} not found")
    else:
        print("  ✗ data/ directory not found")

    # Check model files
    print("\n📄 Training Scripts:")
    scripts = ["train_thinking.py", "train_talker.py", "inference.py", "rebrand.py"]
    for script in scripts:
        script_path = Path(__file__).parent / script
        if script_path.exists():
            print(f"  ✓ {script}")

            # Check if properly rebranded
            with open(script_path) as f:
                content = f.read()
                if "Zen1" in content:
                    print(f"    ✓ Branded as Zen1")
                elif "Zen1" in content or "Zen1" in content:
                    print(f"    ⚠️  Still has old branding")

                # Check LoRA rank in training scripts
                if script.startswith("train_"):
                    if "r=128" in content:
                        print(f"    ✓ LoRA rank set to 128")
                    else:
                        print(f"    ⚠️  LoRA rank not 128")
        else:
            print(f"  ✗ {script} not found")

    # Memory check
    print("\n💾 System Resources:")
    if torch.cuda.is_available():
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        print(f"  Using Apple Silicon MPS")
    else:
        print(f"  ⚠️  No GPU detected, training will be slow")

    import psutil
    print(f"  RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
    print(f"  Available: {psutil.virtual_memory().available / 1e9:.2f} GB")

    print("\n✅ Setup test complete!")
    print("\nNext steps:")
    print("  1. Run 'make setup' to install any missing dependencies")
    print("  2. Run 'make train-thinking' to train reasoning model")
    print("  3. Run 'make train-talker' to train conversation model")
    print("  4. Run 'make test-thinking' or 'make test-talker' to test")

    return True

if __name__ == "__main__":
    test_setup()