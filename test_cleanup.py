#!/usr/bin/env python3
"""Test script to verify the cleanup was successful"""

import os
import subprocess
from pathlib import Path

def check_no_supra_references():
    """Verify no Supra references remain in the codebase"""
    print("Checking for Supra references...")

    result = subprocess.run(
        ["grep", "-r", "-i", "supra", ".",
         "--exclude-dir=.git",
         "--exclude-dir=*_venv",
         "--exclude-dir=external",
         "--exclude=*.pyc",
         "--exclude=test_cleanup.py"],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print("❌ Found Supra references:")
        print(result.stdout[:500])
        return False
    else:
        print("✅ No Supra references found")
        return True

def verify_directory_structure():
    """Verify the new directory structure exists"""
    print("\nVerifying directory structure...")

    required_dirs = [
        "models/zen-nano",
        "models/zen-omni",
        "models/zen-coder",
        "models/zen-next",
        "models/zen-3d",
        "models/zen1",
        "training/configs",
        "training/data",
        "training/scripts",
        "tools/conversion",
        "tools/deployment",
        "tools/evaluation",
        "docs/models",
        "docs/papers",
        "docs/guides",
        "examples",
        "external",
        "output/adapters",
        "output/checkpoints",
        "output/fused",
    ]

    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} missing")
            all_exist = False

    return all_exist

def check_root_cleanliness():
    """Check that root directory is clean"""
    print("\nChecking root directory cleanliness...")

    allowed_root_files = {
        ".gitignore",
        "Dockerfile",
        "Makefile",
        "README.md",
        "requirements.txt",
        "requirements_finetune.txt",
        "setup.sh",
        "unified_deployment.mk",
        "CLEANUP_SUMMARY.md",
        "test_cleanup.py",
        # Log files (can be removed later)
        "download.log",
        "qwen3_pipeline.log",
    }

    allowed_root_dirs = {
        ".git",
        ".claude",
        "axolotl_venv",
        "zen_venv",
        "mlx_lm_lora",
        "base-models",
        "gym-output",
        # New organized directories
        "models",
        "training",
        "tools",
        "docs",
        "examples",
        "external",
        "output",
        "modelfiles",
        "scripts",  # Keep legacy scripts dir for now
    }

    root_items = set(os.listdir("."))
    unexpected_files = []
    unexpected_dirs = []

    for item in root_items:
        path = Path(item)
        if path.is_file() and item not in allowed_root_files:
            unexpected_files.append(item)
        elif path.is_dir() and item not in allowed_root_dirs:
            unexpected_dirs.append(item)

    if unexpected_files:
        print(f"⚠️  Unexpected files in root: {unexpected_files}")
    else:
        print("✅ No unexpected files in root")

    if unexpected_dirs:
        print(f"⚠️  Unexpected directories in root: {unexpected_dirs}")
    else:
        print("✅ No unexpected directories in root")

    return len(unexpected_files) == 0 and len(unexpected_dirs) == 0

def main():
    """Run all verification tests"""
    print("=" * 60)
    print("Zen Repository Cleanup Verification")
    print("=" * 60)

    tests = [
        check_no_supra_references(),
        verify_directory_structure(),
        check_root_cleanliness(),
    ]

    print("\n" + "=" * 60)
    if all(tests):
        print("✅ ALL TESTS PASSED - Cleanup successful!")
    else:
        print("❌ Some tests failed - Review needed")
    print("=" * 60)

if __name__ == "__main__":
    main()