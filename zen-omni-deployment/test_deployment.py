#!/usr/bin/env python3
"""
Test script for Zen-Omni deployment package
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def check_file_exists(file_path: Path, description: str) -> bool:
    """Check if a file exists and report status"""
    if file_path.exists():
        print(f"✓ {description}: {file_path.name}")
        return True
    else:
        print(f"✗ {description}: {file_path.name} - NOT FOUND")
        return False


def validate_json(file_path: Path) -> Tuple[bool, str]:
    """Validate JSON file format"""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True, "Valid JSON"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def check_variant(variant_name: str, base_dir: Path) -> Dict[str, bool]:
    """Check all files for a specific variant"""
    print(f"\n{'='*50}")
    print(f"Checking {variant_name.upper()} variant")
    print('='*50)

    variant_dir = base_dir / "models" / f"zen-omni-{variant_name}"
    results = {}

    # Check required files
    required_files = [
        ("README.md", "Model card"),
        ("config.json", "Model configuration"),
    ]

    for filename, description in required_files:
        file_path = variant_dir / filename
        exists = check_file_exists(file_path, description)
        results[filename] = exists

        # Additional validation for JSON files
        if filename.endswith('.json') and exists:
            valid, message = validate_json(file_path)
            print(f"  └─ {message}")
            results[f"{filename}_valid"] = valid

    # Check generated Python files (will be created by deploy script)
    python_files = [
        "configuration_zen_omni.py",
        "modeling_zen_omni.py",
        "tokenizer_config.json",
        "generation_config.json"
    ]

    print(f"\nFiles to be generated during deployment:")
    for filename in python_files:
        print(f"  • {filename}")

    return results


def check_paper(base_dir: Path) -> bool:
    """Check paper files"""
    print(f"\n{'='*50}")
    print("Checking PAPER files")
    print('='*50)

    paper_dir = base_dir / "paper"
    tex_file = paper_dir / "zen-omni.tex"
    makefile = paper_dir / "Makefile"

    tex_exists = check_file_exists(tex_file, "LaTeX paper")
    make_exists = check_file_exists(makefile, "Build script")

    if tex_exists:
        # Check LaTeX file size and basic structure
        content = tex_file.read_text()
        lines = len(content.split('\n'))
        print(f"  └─ Paper length: {lines} lines")

        # Check for key sections
        sections = ["Introduction", "Architecture", "Experiments", "Conclusion"]
        for section in sections:
            if f"\\section{{{section}}}" in content:
                print(f"  └─ Section found: {section}")

    return tex_exists and make_exists


def check_deployment_files(base_dir: Path) -> bool:
    """Check deployment infrastructure files"""
    print(f"\n{'='*50}")
    print("Checking DEPLOYMENT files")
    print('='*50)

    files = [
        ("deploy_to_hf.py", "Deployment script"),
        ("requirements.txt", "Python dependencies"),
        ("README.md", "Main documentation"),
        ("test_deployment.py", "Test script (this file)")
    ]

    all_exist = True
    for filename, description in files:
        file_path = base_dir / filename
        exists = check_file_exists(file_path, description)
        all_exist = all_exist and exists

        # Check if deployment script is executable
        if filename == "deploy_to_hf.py" and exists:
            content = file_path.read_text()
            if content.startswith("#!/usr/bin/env python"):
                print(f"  └─ Has shebang line")

    return all_exist


def main():
    """Main test function"""
    print("="*60)
    print(" Zen-Omni Deployment Package Validation")
    print("="*60)

    base_dir = Path(__file__).parent
    print(f"\nBase directory: {base_dir}")

    # Check each variant
    variants = ["thinking", "talking", "captioner"]
    variant_results = {}

    for variant in variants:
        results = check_variant(variant, base_dir)
        variant_results[variant] = all(results.values())

    # Check paper
    paper_ok = check_paper(base_dir)

    # Check deployment files
    deploy_ok = check_deployment_files(base_dir)

    # Summary
    print(f"\n{'='*60}")
    print(" VALIDATION SUMMARY")
    print('='*60)

    all_ok = True

    print("\nModel Variants:")
    for variant, ok in variant_results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} zen-omni-{variant}")
        all_ok = all_ok and ok

    print(f"\nPaper: {'✓' if paper_ok else '✗'}")
    print(f"Deployment Files: {'✓' if deploy_ok else '✗'}")

    all_ok = all_ok and paper_ok and deploy_ok

    if all_ok:
        print("\n✅ All validation checks PASSED!")
        print("\nNext steps:")
        print("1. Review model cards and configurations")
        print("2. Build the paper: cd paper && make all")
        print("3. Deploy to HuggingFace: python deploy_to_hf.py --all")
        return 0
    else:
        print("\n❌ Some validation checks FAILED")
        print("Please review the errors above and fix missing files")
        return 1


if __name__ == "__main__":
    sys.exit(main())