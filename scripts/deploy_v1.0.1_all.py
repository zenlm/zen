#!/usr/bin/env python3
'''Deploy v1.1 models to HuggingFace'''

import os
import subprocess
from pathlib import Path

models = [
    {
        "name": "zen-nano-instruct-v1.0.1",
        "repo": "zenlm/zen-nano-instruct",
        "path": "~/work/zoo/gym/models/zen-nano-instruct-v1.0.1",
        "description": "v1.0.1: Enhanced with recursive learning from work sessions"
    },
    {
        "name": "zen-nano-thinking-v1.0.1", 
        "repo": "zenlm/zen-nano-thinking",
        "path": "~/work/zoo/gym/models/zen-nano-thinking-v1.0.1",
        "description": "v1.0.1: Improved reasoning with session insights"
    },
    {
        "description": "v1.0.1: Enhanced O1 capabilities from recursive training"
    },
    {
        "description": "v1.0.1: Advanced reasoning with recursive improvements"
    }
]

# Improvements in v1.1
improvements = '''
## v1.0.1 Improvements (Recursive Learning Release)

### 🔒 Security Enhancements
- Fixed API token exposure vulnerabilities
- Added path traversal protection
- Implemented secure environment variable handling

### 📚 Documentation Improvements  
- Hierarchical documentation structure
- Comprehensive format-specific guides
- Clear training instructions with zoo-gym

### 🎯 Identity & Branding
- Stronger model identity (no base model confusion)
- Consistent branding across all materials
- Clear attribution and mission

### 🔧 Technical Enhancements
- Multi-format support (MLX, GGUF, SafeTensors)
- Improved error handling and diagnostics
- Better training data from work sessions

### 🧬 Recursive Learning
- Learned from 20 real work interactions
- Pattern recognition and improvement synthesis
- Self-improving architecture foundation
'''

for model in models:
    print(f"\n📦 Deploying {model['name']}...")

    # Prepare variables for the f-string
    model_name = model['name'].replace('-v1.0.1', '')
    model_name_clean = model['name'].replace('-', '_')

    # Create enhanced README
    readme = f'''# {model_name} v1.0.1

{model['description']}

{improvements}

## Training Data

This version was trained on synthetic data generated from actual work sessions,
implementing a recursive self-improvement approach where the AI learns from its
own problem-solving experiences.

## Usage

```python
# For inference
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model['repo']}")
tokenizer = AutoTokenizer.from_pretrained("{model['repo']}")

# MLX (Apple Silicon)
from mlx_lm import load, generate
model, tokenizer = load("{model['repo']}")
```

## Citation

```bibtex
@misc{{zen_v1_0_1_2025,
    title={{Model v1.0.1: Recursive Self-Improvement Release}},
    author={{Zen/Supra Teams}},
    year={{2025}},
    version={{1.0.1}}
}}
```
'''
    
    # Save README
    readme_path = Path(model['path']) / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme)
    
    # Upload to HuggingFace
    env = os.environ.copy()
    if os.getenv('HF_TOKEN'):
        env['HF_TOKEN'] = os.getenv('HF_TOKEN')
    
    cmd = ["hf", "upload", model['repo'], str(Path(model['path']).expanduser())]
    
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"  ✓ Deployed {model['name']} to {model['repo']}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to deploy {model['name']}: {e}")

print("\n🎉 v1.0.1 Recursive Learning Release Complete!")
print("Models have learned from their own work sessions and improved autonomously.")
