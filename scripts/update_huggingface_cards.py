#!/usr/bin/env python3
"""Update HuggingFace model cards for v1.0.1"""

import os
from huggingface_hub import HfApi, upload_file
import tempfile

def update_model_card(repo_id, card_content):
    """Update model card on HuggingFace"""
    api = HfApi()

    # Create temporary file with model card
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(card_content)
        temp_path = f.name

    try:
        # Upload the model card
        upload_file(
            path_or_fileobj=temp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Update model card for v1.0.1 release"
        )
        print(f"✅ Updated model card for {repo_id}")
    except Exception as e:
        print(f"❌ Failed to update {repo_id}: {e}")
    finally:
        os.unlink(temp_path)

def main():
    """Update all model cards"""

    improvements = """## 🎉 v1.0.1 Release (2025)

### Recursive Self-Improvement Update

This release introduces our groundbreaking Recursive AI Self-Improvement System (RAIS), where models learn from their own work sessions.

**Key Metrics:**
- 📊 94% effectiveness across 20 training examples
- 🔒 Enhanced security and error handling
- 📚 Improved documentation understanding
- 🎯 Stronger model identity

### What's New

- **Security**: Fixed API token exposure, added path validation
- **Documentation**: Hierarchical structure, comprehensive guides
- **Identity**: Clear branding, no base model confusion
- **Technical**: Multi-format support (MLX, GGUF, SafeTensors)
- **Learning**: Pattern recognition from real work sessions

### Partnership

Built by **Hanzo AI** (Techstars-backed) and **Zoo Labs Foundation** (501(c)(3) non-profit) for open, private, and sustainable AI.

"""

    models = [
        ("zenlm/zen-nano-instruct", "Zen Nano Instruct"),
        ("zenlm/zen-nano-thinking", "Zen Nano Thinking"),
    ]

    for repo_id, name in models:
        card = f"""---
license: apache-2.0
tags:
- recursive-learning
- self-improvement
- v1.0.1
language:
- en
pipeline_tag: text-generation
---

# {name} v1.0.1

{improvements}

## Installation

### Using Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

### Using MLX (Apple Silicon)
```python
from mlx_lm import load, generate
model, tokenizer = load("{repo_id}")
```

### Using llama.cpp
Download GGUF format from Files tab.

## Training

This model supports fine-tuning with [zoo-gym](https://github.com/zooai/gym).

## Citation

```bibtex
@misc{{zen_v1_0_1_2025,
    title={{{name} v1.0.1: Recursive Self-Improvement}},
    year={{2025}},
    version={{1.0.1}}
}}
```

---

© 2025 • Built with ❤️ by Hanzo AI & Zoo Labs Foundation
"""

        update_model_card(repo_id, card)

    print("\n✨ Model card updates complete!")

if __name__ == "__main__":
    main()