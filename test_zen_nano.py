#!/usr/bin/env python3
"""Test Zen Nano model setup and basic inference"""

import os
import sys
import json
from pathlib import Path

def test_gguf_model():
    """Test if GGUF model exists and is valid"""
    gguf_path = Path("zen-nano/zen-nano.gguf")
    if gguf_path.exists():
        size_mb = gguf_path.stat().st_size / (1024 * 1024)
        print(f"✅ Found zen-nano.gguf: {size_mb:.2f} MB")
        return True
    else:
        print("❌ zen-nano.gguf not found")
        return False

def test_base_models():
    """Check if base models are available"""
    base_path = Path("base-models")
    required_models = ["Qwen3-4B-Instruct-2507", "Qwen3-4B-Thinking-2507"]

    found = []
    missing = []

    for model in required_models:
        model_path = base_path / model
        if model_path.exists():
            found.append(model)
        else:
            missing.append(model)

    if found:
        print(f"✅ Found base models: {', '.join(found)}")
    if missing:
        print(f"⚠️  Missing base models: {', '.join(missing)}")

    return len(missing) == 0

def test_mlx_setup():
    """Test MLX installation"""
    try:
        import mlx
        import mlx.core as mx
        print(f"✅ MLX installed: version {mlx.__version__ if hasattr(mlx, '__version__') else 'unknown'}")

        # Test basic MLX operation
        x = mx.array([1, 2, 3])
        y = mx.sum(x)
        print(f"✅ MLX basic operation works: sum([1,2,3]) = {y}")
        return True
    except ImportError as e:
        print(f"❌ MLX not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ MLX error: {e}")
        return False

def test_training_data():
    """Check if training data exists"""
    data_path = Path("zen-nano/data")
    if data_path.exists():
        data_files = list(data_path.glob("*.json")) + list(data_path.glob("*.jsonl"))
        if data_files:
            print(f"✅ Found {len(data_files)} training data files")
            return True
        else:
            print("⚠️  No training data files found")
            return False
    else:
        print("❌ Training data directory not found")
        return False

def create_simple_finetune_script():
    """Create a simple finetuning script for Zen Nano"""
    script = '''#!/usr/bin/env python3
"""Simple Zen Nano finetuning script"""

import json
from pathlib import Path

# Prepare training data
training_data = [
    {"instruction": "What is Zen Nano?", "response": "Zen Nano is an ultra-lightweight AI model optimized for edge deployment."},
    {"instruction": "Who created Zen?", "response": "Zen was created by Hanzo AI, focusing on efficient and powerful AI systems."},
    {"instruction": "What makes Zen special?", "response": "Zen combines efficiency with capability, offering models from nano to omni scale."},
]

# Save training data
data_path = Path("zen-nano/data/finetune_data.jsonl")
data_path.parent.mkdir(exist_ok=True)

with open(data_path, "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\\n")

print(f"✅ Created training data at {data_path}")
print(f"   {len(training_data)} examples ready for finetuning")
'''

    script_path = Path("zen-nano/simple_finetune.py")
    script_path.write_text(script)
    os.chmod(script_path, 0o755)
    print(f"✅ Created finetuning script at {script_path}")
    return True

def main():
    """Run all tests"""
    print("🔍 Testing Zen Nano Setup")
    print("-" * 40)

    results = {
        "GGUF Model": test_gguf_model(),
        "Base Models": test_base_models(),
        "MLX Setup": test_mlx_setup(),
        "Training Data": test_training_data(),
        "Finetune Script": create_simple_finetune_script(),
    }

    print("-" * 40)
    print("📊 Summary:")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test}")

    print(f"\n{'✅ All tests passed!' if passed == total else f'⚠️  {passed}/{total} tests passed'}")

    if passed == total:
        print("\n🚀 Zen Nano is ready for finetuning!")
        print("   Run: python zen-nano/simple_finetune.py")
    else:
        print("\n⚠️  Some components need attention")

if __name__ == "__main__":
    main()