#!/usr/bin/env python3
"""
Publish Zen-1 models to HuggingFace Hub
"""

import os
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, HfFolder

print("""
╔═══════════════════════════════════════════════════════╗
║         PUBLISH ZEN-1 TO HUGGINGFACE                 ║
║              Organization: zenlm                      ║
╚═══════════════════════════════════════════════════════╝
""")

def check_token():
    """Check for HF token"""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if not token:
        # Try to get from HF CLI
        try:
            token = HfFolder.get_token()
        except:
            pass

    if not token:
        print("❌ No HuggingFace token found!")
        print("\n📝 To set token:")
        print("1. Get token at: https://huggingface.co/settings/tokens")
        print("2. Set with: export HF_TOKEN=hf_...")
        print("   Or: huggingface-cli login")
        return None

    print("✅ HuggingFace token found")
    return token

def prepare_model_files():
    """Prepare model files for upload"""

    # Check what we have
    locations = {
        "gym_model": Path("gym-output/model"),
        "zen1_instruct": Path("zen-1/instruct"),
        "zen1_thinking": Path("zen-1/thinking"),
    }

    available = {}
    for name, path in locations.items():
        if path.exists():
            available[name] = path
            print(f"✅ Found {name}: {path}")
        else:
            print(f"⚠️  Not found: {path}")

    return available

def create_model_card(model_type="base"):
    """Create model card for HuggingFace"""

    if model_type == "instruct":
        base_model = "Qwen3-4B-Instruct-2507"
        description = "Direct instruction following variant"
    elif model_type == "thinking":
        base_model = "Qwen3-4B-Thinking-2507"
        description = "Chain-of-thought reasoning variant"
    else:
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        description = "Base fine-tuned model"

    return f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - zen
  - hanzo
  - fine-tuned
  - mcp
  - {"instruct" if model_type == "instruct" else "thinking" if model_type == "thinking" else "general"}
language:
  - en
  - code
pipeline_tag: text-generation
---

# Zen-1{"-" + model_type.title() if model_type != "base" else ""}

{description} of Zen-1, fine-tuned for advanced language understanding.

## Key Features

- **Enhanced Reasoning**: Improved chain-of-thought capabilities
- **Code Generation**: Strong performance on programming tasks
- **Instruction Following**: Precise adherence to user instructions
- **Multi-turn Dialogue**: Coherent conversation handling
- **Technical Knowledge**: Deep understanding of ML/AI concepts

## Installation

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-1{"-" + model_type if model_type != "base" else ""}")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-1{"-" + model_type if model_type != "base" else ""}")

# Generate
inputs = tokenizer("Explain gradient descent", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### With Ollama

```bash
ollama run zenlm/zen-1{"-" + model_type if model_type != "base" else ""}
```

## Training Details

- **Method**: {"LoRA fine-tuning" if model_type == "base" else "MLX fine-tuning"}
- **Hardware**: Apple Silicon (M-series)
- **Base Model**: {base_model}
- **Training Data**: High-quality instruction and reasoning datasets

## License

Apache 2.0

## Citation

```bibtex
@misc{{zen1-2024,
  title={{Zen-1{"-" + model_type.title() if model_type != "base" else ""}: Advanced Language Model}},
  author={{Zen Team}},
  year={{2024}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/zenlm/zen-1{"-" + model_type if model_type != "base" else ""}}}
}}
```
"""

def upload_model(api, model_path, repo_id, model_type="base", token=None):
    """Upload model to HuggingFace"""

    try:
        # Create repo
        print(f"\n📦 Creating repository: {repo_id}")
        create_repo(
            repo_id,
            token=token,
            repo_type="model",
            exist_ok=True,
            private=False
        )

        # Add model card
        model_card = create_model_card(model_type)
        model_card_path = Path(model_path) / "README.md"
        model_card_path.write_text(model_card)
        print(f"📝 Added model card")

        # Upload
        print(f"📤 Uploading to {repo_id}...")
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload Zen-1{'-' + model_type.title() if model_type != 'base' else ''} model"
        )

        print(f"✅ Uploaded to https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    # Check token
    token = check_token()
    if not token:
        return

    api = HfApi(token=token)

    # Find available models
    available = prepare_model_files()

    if not available:
        print("\n❌ No models found to upload!")
        return

    print("\n📊 Available models to upload:")
    options = list(available.keys())
    for i, name in enumerate(options, 1):
        print(f"{i}. {name}")

    print(f"{len(options)+1}. Upload all")

    choice = input(f"\nChoice [1]: ").strip() or "1"

    # Upload based on choice
    uploaded = []

    if choice == str(len(options)+1):
        # Upload all
        uploads = [
            ("gym_model", "gym-output/model", "zenlm/zen-1", "base"),
            ("zen1_instruct", "zen-1/instruct", "zenlm/zen-1-instruct", "instruct"),
            ("zen1_thinking", "zen-1/thinking", "zenlm/zen-1-thinking", "thinking"),
        ]
    else:
        # Upload selected
        idx = int(choice) - 1
        selected = options[idx]

        if selected == "gym_model":
            uploads = [("gym_model", available[selected], "zenlm/zen-1", "base")]
        elif selected == "zen1_instruct":
            uploads = [("zen1_instruct", available[selected], "zenlm/zen-1-instruct", "instruct")]
        elif selected == "zen1_thinking":
            uploads = [("zen1_thinking", available[selected], "zenlm/zen-1-thinking", "thinking")]

    # Perform uploads
    for name, path, repo_id, model_type in uploads:
        if name in available:
            print(f"\n🚀 Uploading {name}...")
            if upload_model(api, path, repo_id, model_type, token):
                uploaded.append(repo_id)

    # Summary
    if uploaded:
        print("\n" + "="*60)
        print("🎉 Upload Complete!")
        print("="*60)
        print("\n📦 Published models:")
        for repo_id in uploaded:
            print(f"  • https://huggingface.co/{repo_id}")

        print("\n🚀 To use:")
        print("  from transformers import pipeline")
        print(f"  pipe = pipeline('text-generation', '{uploaded[0]}')")
        print("  print(pipe('How do I use hanzo-mcp?'))")
    else:
        print("\n⚠️  No models uploaded")

if __name__ == "__main__":
    main()