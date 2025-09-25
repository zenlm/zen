#!/usr/bin/env python3
"""Upload Zen Nano 4B Thinking 4-bit to Hugging Face"""

from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path

# Configuration
REPO_ID = "zenlm/zen-nano-thinking-4bit"
MODEL_PATH = "models/zen-nano-4b-thinking-mlx-q4"

print("🧠 Uploading Zen Nano 4B Thinking (4-bit) to Hugging Face")
print(f"   Repository: {REPO_ID}")
print(f"   Model path: {MODEL_PATH}")

# Initialize API
api = HfApi()

# Create or get repository
try:
    repo_url = create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")
    print(f"✅ Repository ready: {repo_url}")
except Exception as e:
    print(f"❌ Error creating repository: {e}")
    exit(1)

# Upload all files
print("📤 Uploading model files...")
try:
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload Zen Nano 4B Thinking 4-bit MLX model - Advanced reasoning with ultra-efficient inference"
    )
    print(f"✅ Model uploaded successfully!")
    print(f"   View at: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"❌ Error uploading: {e}")
    exit(1)