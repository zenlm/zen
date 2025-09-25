#!/usr/bin/env python3
"""
Zen1 Rebranding Script
Updates all files with Zen1 branding
"""

import os
import re
from pathlib import Path


def update_branding(file_path, replacements):
    """Update branding in a single file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        original = content
        for old, new in replacements.items():
            # Case-insensitive replacement while preserving some case patterns
            content = re.sub(f"Qwen3-Omni", "Zen1", content, flags=re.IGNORECASE)
            content = re.sub(f"Qwen-Omni", "Zen1", content, flags=re.IGNORECASE)
            content = re.sub(f"qwen3-omni", "zen1", content, flags=re.IGNORECASE)
            content = re.sub(f"qwen-omni", "zen1", content, flags=re.IGNORECASE)
            content = re.sub(f"Zen-1", "Zen1", content)
            content = re.sub(f"zen-1", "zen1", content)

        # Specific replacements
        content = content.replace("Qwen Team", "Zen Team")
        content = content.replace("Qwen team", "Zen team")
        content = content.replace("Alibaba", "Zen AI")

        if content != original:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
    return False


def update_lora_rank(file_path, new_rank=128):
    """Update LoRA rank in Python files"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        original = content
        # Update LoRA rank
        content = re.sub(r'r=\d+,\s*#\s*Rank', f'r={new_rank},  # Rank', content)
        content = re.sub(r'lora_r:\s*int\s*=\s*\d+', f'lora_r: int = {new_rank}', content)
        content = re.sub(r'--lora_r\s+\d+', f'--lora_r {new_rank}', content)

        # Update alpha to 2x rank
        content = re.sub(r'lora_alpha=\d+', f'lora_alpha={new_rank * 2}', content)
        content = re.sub(r'lora_alpha:\s*int\s*=\s*\d+', f'lora_alpha: int = {new_rank * 2}', content)

        if content != original:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"Error updating LoRA rank in {file_path}: {e}")
    return False


def main():
    """Run rebranding"""

    print("🎨 Zen1 Rebranding Tool")
    print("=" * 50)

    # Define replacements
    replacements = {
        "Qwen3-Omni": "Zen1",
        "Qwen-Omni": "Zen1",
        "qwen3-omni": "zen1",
        "qwen-omni": "zen1",
    }

    # Get all files to update
    zen1_dir = Path(__file__).parent
    files_to_update = []

    # Python files
    files_to_update.extend(zen1_dir.glob("**/*.py"))
    # Markdown files
    files_to_update.extend(zen1_dir.glob("**/*.md"))
    # YAML files
    files_to_update.extend(zen1_dir.glob("**/*.yaml"))
    files_to_update.extend(zen1_dir.glob("**/*.yml"))
    # Makefiles
    files_to_update.extend(zen1_dir.glob("**/Makefile"))
    # Shell scripts
    files_to_update.extend(zen1_dir.glob("**/*.sh"))

    updated_files = []

    # Update branding
    print("\n📝 Updating branding...")
    for file_path in files_to_update:
        if file_path.name == "rebrand.py":
            continue  # Skip this script

        if update_branding(file_path, replacements):
            updated_files.append(file_path)
            print(f"  ✓ {file_path.relative_to(zen1_dir)}")

    # Update LoRA rank in Python files
    print("\n🔧 Updating LoRA rank to 128...")
    python_files = list(zen1_dir.glob("**/*.py"))
    for file_path in python_files:
        if file_path.name == "rebrand.py":
            continue

        if update_lora_rank(file_path, 128):
            print(f"  ✓ {file_path.relative_to(zen1_dir)}")

    print(f"\n✅ Rebranding complete!")
    print(f"   Updated {len(updated_files)} files")
    print(f"\n🚀 Your Zen1 models are ready!")
    print(f"   Run 'make setup' to install dependencies")
    print(f"   Run 'make train-thinking' for reasoning model")
    print(f"   Run 'make train-talker' for conversation model")


if __name__ == "__main__":
    main()