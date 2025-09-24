#!/usr/bin/env python3
"""
Verify integrity of downloaded Qwen3 models.
Calculates and stores checksums for model files.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List
import concurrent.futures
import time

BASE_DIR = Path("/Users/z/work/zen/base-models")
CHECKSUM_FILE = BASE_DIR / "checksums.json"

# File extensions to checksum
MODEL_EXTENSIONS = [".safetensors", ".bin", ".pt", ".model", ".gguf"]
CONFIG_EXTENSIONS = [".json", ".txt", ".yaml", ".yml"]


def calculate_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def calculate_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file (faster, for quick checks)."""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def get_file_info(filepath: Path) -> Dict:
    """Get file information including size and checksums."""
    size = filepath.stat().st_size
    print(f"  Calculating checksums for {filepath.name} ({size / (1024**2):.1f} MB)...")
    
    return {
        "path": str(filepath.relative_to(BASE_DIR)),
        "size": size,
        "md5": calculate_md5(filepath),
        "sha256": calculate_sha256(filepath) if size < 1024**3 else None  # Skip SHA256 for files > 1GB
    }


def verify_model_directory(model_dir: Path) -> Dict:
    """Verify all files in a model directory."""
    if not model_dir.exists():
        return {"error": f"Directory not found: {model_dir}"}
    
    print(f"\n🔍 Verifying {model_dir.name}...")
    
    # Find all relevant files
    model_files = []
    config_files = []
    
    for ext in MODEL_EXTENSIONS:
        model_files.extend(model_dir.glob(f"*{ext}"))
    
    for ext in CONFIG_EXTENSIONS:
        config_files.extend(model_dir.glob(f"*{ext}"))
    
    # Calculate checksums
    all_files = model_files + config_files
    file_info = []
    
    # Use thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(get_file_info, f) for f in all_files]
        for future in concurrent.futures.as_completed(futures):
            try:
                info = future.result()
                file_info.append(info)
            except Exception as e:
                print(f"  ⚠️  Error processing file: {e}")
    
    # Calculate total size
    total_size = sum(f["size"] for f in file_info)
    
    result = {
        "model": model_dir.name,
        "total_size": total_size,
        "total_size_gb": total_size / (1024**3),
        "model_files": len(model_files),
        "config_files": len(config_files),
        "files": sorted(file_info, key=lambda x: x["path"])
    }
    
    print(f"  ✅ {len(model_files)} model files, {len(config_files)} config files")
    print(f"  📊 Total size: {total_size / (1024**3):.2f} GB")
    
    return result


def save_checksums(checksums: Dict):
    """Save checksums to JSON file."""
    with open(CHECKSUM_FILE, "w") as f:
        json.dump(checksums, f, indent=2)
    print(f"\n💾 Checksums saved to {CHECKSUM_FILE}")


def load_checksums() -> Dict:
    """Load existing checksums."""
    if CHECKSUM_FILE.exists():
        with open(CHECKSUM_FILE, "r") as f:
            return json.load(f)
    return {}


def compare_checksums(current: Dict, stored: Dict) -> bool:
    """Compare current checksums with stored ones."""
    differences = []
    
    for model, data in current.items():
        if model not in stored:
            differences.append(f"New model: {model}")
            continue
        
        stored_data = stored[model]
        
        # Compare file counts
        if data.get("model_files") != stored_data.get("model_files"):
            differences.append(f"{model}: Model file count changed")
        
        # Compare individual files
        current_files = {f["path"]: f for f in data.get("files", [])}
        stored_files = {f["path"]: f for f in stored_data.get("files", [])}
        
        for path, file_info in current_files.items():
            if path not in stored_files:
                differences.append(f"{model}: New file {path}")
            elif file_info["md5"] != stored_files[path]["md5"]:
                differences.append(f"{model}: Checksum mismatch for {path}")
    
    if differences:
        print("\n⚠️  Differences found:")
        for diff in differences:
            print(f"  - {diff}")
        return False
    
    print("\n✅ All checksums match!")
    return True


def main():
    """Main verification process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Qwen3 model integrity")
    parser.add_argument(
        "models",
        nargs="*",
        help="Specific models to verify (default: all)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save checksums after verification"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with stored checksums"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check (file existence and size only)"
    )
    
    args = parser.parse_args()
    
    # Find model directories
    if args.models:
        model_dirs = [BASE_DIR / m for m in args.models]
    else:
        model_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    if not model_dirs:
        print("❌ No model directories found in", BASE_DIR)
        return 1
    
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"🔍 Found {len(model_dirs)} model directories")
    
    # Quick check mode
    if args.quick:
        print("\n⚡ Quick verification mode (existence and size only)")
        for model_dir in model_dirs:
            if not model_dir.exists():
                print(f"❌ {model_dir.name}: Not found")
                continue
            
            size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            model_files = list(model_dir.glob("*.safetensors")) + \
                         list(model_dir.glob("*.bin")) + \
                         list(model_dir.glob("*.pt"))
            
            print(f"✅ {model_dir.name}: {len(model_files)} model files, {size / (1024**3):.2f} GB")
        return 0
    
    # Full verification
    start_time = time.time()
    checksums = {}
    
    for model_dir in model_dirs:
        result = verify_model_directory(model_dir)
        if "error" not in result:
            checksums[model_dir.name] = result
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Verification completed in {elapsed:.1f} seconds")
    
    # Compare with stored checksums
    if args.compare:
        stored = load_checksums()
        if stored:
            compare_checksums(checksums, stored)
        else:
            print("⚠️  No stored checksums found")
    
    # Save checksums
    if args.save:
        save_checksums(checksums)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Verification Summary:")
    print("="*60)
    
    total_size = 0
    total_files = 0
    
    for model_name, data in checksums.items():
        if "error" in data:
            print(f"❌ {model_name}: {data['error']}")
        else:
            size_gb = data["total_size_gb"]
            total_size += size_gb
            total_files += data["model_files"] + data["config_files"]
            print(f"✅ {model_name}:")
            print(f"   - Model files: {data['model_files']}")
            print(f"   - Config files: {data['config_files']}")
            print(f"   - Total size: {size_gb:.2f} GB")
    
    print(f"\n📈 Total: {total_files} files, {total_size:.2f} GB")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())