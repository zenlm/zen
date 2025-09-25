#!/usr/bin/env python3
"""
Test Supra Nexus O1 models
Minimal inference validation
"""

import sys
from pathlib import Path

def test_local_model(model_path: Path) -> None:
    """Test local model with transformers."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading: {model_path.name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load with reduced memory for testing
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Simple test
        prompt = "What is 2+2?"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Model loaded successfully")
        print(f"  Prompt: {prompt}")
        print(f"  Response: {response[:100]}...")
        
    except ImportError:
        print("✗ transformers not installed")
        print("  Install: pip install transformers torch")
    except Exception as e:
        print(f"✗ Error: {e}")

def test_mlx_model(model_path: Path) -> None:
    """Test model with MLX."""
    try:
        from mlx_lm import load, generate
        
        print(f"Loading with MLX: {model_path.name}")
        model, tokenizer = load(str(model_path))
        
        # Simple test
        prompt = "What is 2+2?"
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=50,
            temp=0.7
        )
        
        print(f"✓ MLX model loaded successfully")
        print(f"  Prompt: {prompt}")
        print(f"  Response: {response[:100]}...")
        
    except ImportError:
        print("✗ MLX not installed")
        print("  Install: pip install mlx mlx-lm")
    except Exception as e:
        print(f"✗ Error: {e}")

def test_remote_model(repo_id: str) -> None:
    """Test remote model from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
        
        print(f"Testing remote: {repo_id}")
        
        # Check if model exists
        try:
            cache_dir = snapshot_download(
                repo_id=repo_id,
                cache_dir="./model_cache",
                allow_patterns=["config.json"]
            )
            print(f"✓ Model accessible at: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"✗ Model not accessible: {e}")
            
    except ImportError:
        print("✗ huggingface-hub not installed")

def main():
    """Test workflow."""
    print("Supra Nexus O1 Model Testing")
    print("=" * 50)
    
    # Test paths
    local_models = {
        "thinking": Path("/Users/z/work/supra/o1/models/supra-nexus-o1-thinking-fused"),
        "instruct": Path("/Users/z/work/supra/o1/models/supra-nexus-o1-instruct-fused")
    }
    
    remote_models = [
        "supra-nexus/supra-nexus-o1-thinking",
        "supra-nexus/supra-nexus-o1-instruct"
    ]
    
    # Test local models
    print("\nLocal Model Tests:")
    print("-" * 50)
    for name, path in local_models.items():
        if path.exists():
            test_local_model(path)
            print()
    
    # Test remote models
    print("\nRemote Model Tests:")
    print("-" * 50)
    for repo_id in remote_models:
        test_remote_model(repo_id)
        print()
    
    print("=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    main()