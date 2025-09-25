#!/usr/bin/env python3
"""
Secure Supra Nexus O1 Deployment Script
Fixed security vulnerabilities and improved error handling
"""

import os
import sys
import json
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Dict, List

def validate_repo_name(repo_name: str) -> bool:
    """Validate repository name format"""
    import re
    pattern = r'^[a-zA-Z0-9][a-zA-Z0-9-_\.]*\/[a-zA-Z0-9][a-zA-Z0-9-_\.]*$'
    return bool(re.match(pattern, repo_name))

def validate_path(path: str) -> Path:
    """Validate and resolve path securely"""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")
    return resolved

def run_command(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Run command with proper error handling and no shell injection"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Handle errors manually
            **kwargs
        )
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {result.stderr}")
        return result
    except Exception as e:
        print(f"Failed to execute command: {e}")
        sys.exit(1)

def create_model_card(model_name: str, model_type: str = "thinking") -> str:
    """Create secure model card content"""
    
    # Sanitize inputs
    model_name = model_name.replace('..', '').replace('/', '_')
    model_type = model_type.lower()
    
    if model_type not in ["thinking", "instruct"]:
        model_type = "thinking"
    
    type_name = "Thinking Model" if model_type == "thinking" else "Instruct Model"
    capability = "transparent reasoning" if model_type == "thinking" else "direct responses"
    
    return f"""---
license: apache-2.0
language:
- en
tags:
- supra-nexus
- reasoning
- {model_type}
pipeline_tag: text-generation
---

# {model_name}

Advanced {type_name} with {capability} by **Supra Foundation LLC**.

## Model Details

- **Architecture**: 4B parameters based on Qwen3
- **Context**: 32K tokens
- **License**: Apache 2.0
- **Year**: 2025

## Available Formats

### MLX (Apple Silicon)
```bash
pip install mlx mlx-lm
```

### GGUF (llama.cpp)
```bash
huggingface-cli download {model_name} --include "*.gguf"
```

### Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
```

## Citation

```bibtex
@misc{{supranexus2025,
  title={{Supra Nexus O1: Advanced Reasoning AI}},
  author={{Supra Foundation LLC}},
  year={{2025}},
  publisher={{HuggingFace}}
}}
```

---

Built by Supra Foundation LLC • 2025
"""

def upload_model(model_path: Path, repo_name: str, model_type: str = "thinking") -> bool:
    """Upload model to HuggingFace with security checks"""
    
    # Validate inputs
    if not validate_repo_name(repo_name):
        print(f"Invalid repository name format: {repo_name}")
        return False
    
    try:
        model_path = validate_path(str(model_path))
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Invalid model path: {e}")
        return False
    
    print(f"Uploading {model_path.name} to {repo_name}...")
    
    # Create repository
    result = run_command(["hf", "repo", "create", repo_name])
    if result.returncode == 0:
        print(f"Created repository: {repo_name}")
    elif "already exists" in result.stderr.lower():
        print(f"Repository already exists: {repo_name}")
    else:
        return False
    
    # Create and upload README
    readme_path = model_path / "README.md"
    model_card = create_model_card(repo_name, model_type)
    
    try:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card)
    except IOError as e:
        print(f"Failed to write README: {e}")
        return False
    
    # Upload files (no shell, direct command)
    upload_cmd = ["hf", "upload", repo_name, str(model_path)]
    result = run_command(upload_cmd)
    
    if result.returncode == 0:
        print(f"✅ Successfully uploaded {repo_name}")
        return True
    else:
        print(f"❌ Failed to upload {repo_name}")
        return False

def main():
    """Main deployment function"""
    
    # Get base path
    base_dir = Path("/Users/z/work/supra/o1/models")
    
    # Models to deploy
    models = [
        {
            "path": base_dir / "supra-nexus-o1-thinking-fused",
            "repo": "Supra-Nexus/supra-nexus-o1-thinking",
            "type": "thinking"
        },
        {
            "path": base_dir / "supra-nexus-o1-instruct-fused",
            "repo": "Supra-Nexus/supra-nexus-o1-instruct",
            "type": "instruct"
        }
    ]
    
    # Check HF CLI
    result = run_command(["which", "hf"])
    if result.returncode != 0:
        print("HuggingFace CLI not found. Please install with: pip install huggingface-hub")
        sys.exit(1)
    
    # Deploy models
    success_count = 0
    for model in models:
        if upload_model(model["path"], model["repo"], model["type"]):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Deployment Summary: {success_count}/{len(models)} models uploaded")
    print(f"{'='*60}")
    
    if success_count == len(models):
        print("🎉 All models deployed successfully!")
        return 0
    else:
        print("⚠️  Some models failed to deploy")
        return 1

if __name__ == "__main__":
    sys.exit(main())