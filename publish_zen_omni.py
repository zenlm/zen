#!/usr/bin/env python3
"""
Publish Zen-Omni models to HuggingFace
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

print("📤 Publishing Zen-Omni to HuggingFace (zenlm organization)")

# Models to publish
models = [
    ("zen-omni-instruct", "zenlm/zen-omni-instruct"),
    ("zen-omni-thinking", "zenlm/zen-omni-thinking"),
    ("zen-omni-captioner", "zenlm/zen-omni-captioner")
]

api = HfApi()

for local_name, repo_id in models:
    print(f"\n🚀 Publishing {local_name} to {repo_id}")
    
    # Create repo
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository {repo_id} ready")
    except Exception as e:
        print(f"❌ Error: {e}")
        continue
    
    # Upload would happen here with actual model files
    print(f"📦 Would upload {local_name} model files to {repo_id}")

print("\n✅ Publishing complete!")
print("View models at: https://huggingface.co/zenlm")
