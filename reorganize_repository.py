#!/usr/bin/env python3
"""
Repository Reorganization Script for Supra-Nexus/o1
Transforms the chaotic repository into a clean, professional structure.
"""

import os
import shutil
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class RepositoryReorganizer:
    def __init__(self, repo_path="/Users/z/work/zen"):
        self.repo_path = Path(repo_path)
        self.backup_path = self.repo_path / "backup_before_reorganization"

    def create_directory_structure(self):
        """Create the new directory structure"""
        directories = [
            # Source directories
            "src/zen/models",
            "src/zen/training",
            "src/zen/inference",
            "src/zen/deployment",
            "src/zen/conversion",
            "src/zen/utils",
            "src/gym_bridge",

            # Scripts directories
            "scripts/train",
            "scripts/deploy",
            "scripts/convert",
            "scripts/benchmark",

            # Config directories
            "configs/models",
            "configs/training",
            "configs/deployment",

            # Documentation directories
            "docs/architecture",
            "docs/guides",
            "docs/api",
            "docs/benchmarks",

            # Data directories
            "data/raw",
            "data/processed",
            "data/datasets",

            # Model directories
            "models/checkpoints",
            "models/pretrained",
            "models/finetuned",
            "models/exports/gguf",
            "models/exports/mlx",
            "models/exports/onnx",
            "models/adapters",

            # Test directories
            "tests/unit",
            "tests/integration",
            "tests/fixtures",

            # Example directories
            "examples/notebooks",
            "examples/scripts",

            # Tool directories
            "tools/setup",
            "tools/ci",
            "tools/monitoring",

            # GitHub directories
            ".github/workflows",
            ".github/ISSUE_TEMPLATE",
        ]

        for directory in directories:
            dir_path = self.repo_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created: {directory}")

    def move_files(self):
        """Move files to their new locations"""
        file_migrations = {
            # Documentation files
            "MLX_GUIDE.md": "docs/guides/mlx_guide.md",
            "TRAINING.md": "docs/guides/training.md",
            "DEPLOYMENT_COMPLETE.md": "docs/architecture/deployment.md",
            "ORGANIZATION.md": "docs/architecture/organization.md",
            "HUGGINGFACE_ORG_PROFILE.md": "docs/guides/huggingface_profile.md",
            "huggingface_org_readme.md": "docs/guides/huggingface.md",
            "SUPRA_NEXUS_DEPLOYMENT.md": "docs/architecture/supra_deployment.md",
            "UPDATE_ORG_REFS.md": "docs/guides/org_references.md",

            # Training scripts
            "finetune_zen_nano.py": "scripts/train/finetune_nano.py",
            "finetune_zen_nano_lora.py": "scripts/train/finetune_lora.py",
            "finetune_zen_mlx.py": "scripts/train/finetune_mlx.py",
            "mlx_lora_finetune.py": "scripts/train/mlx_lora_finetune.py",
            "quick_finetune.py": "scripts/train/quick_finetune.py",
            "full_finetune_zen_nano.py": "scripts/train/full_finetune_nano.py",
            "train_pedagogical.py": "scripts/train/train_pedagogical.py",
            "simple_finetune.py": "scripts/train/simple_finetune.py",

            # Deployment scripts
            "deploy_supra_nexus.py": "scripts/deploy/deploy_supra.py",
            "deploy_supra_nexus_complete.sh": "scripts/deploy/deploy_supra_complete.sh",
            "secure_deploy_supra.py": "scripts/deploy/secure_deploy_supra.py",
            "deploy_to_hf.py": "scripts/deploy/deploy_to_hf.py",
            "upload_all_zen_models.py": "scripts/deploy/upload_all_models.py",
            "streamlined_zen_upload.py": "scripts/deploy/streamlined_upload.py",
            "upload_thinking_4bit.py": "scripts/deploy/upload_thinking_4bit.py",

            # Conversion scripts
            "convert_to_gguf.py": "scripts/convert/to_gguf.py",
            "quantize_supra_nexus.py": "scripts/convert/quantize_supra.py",

            # Test files
            "test_zen_nano.py": "tests/integration/test_nano.py",
            "test_zen_nano_final.py": "tests/integration/test_nano_final.py",
            "test_supra_nexus.py": "tests/integration/test_supra.py",
            "verify_complete_formats.py": "tests/unit/test_formats.py",
            "verify_supra_models.py": "tests/unit/test_supra_models.py",

            # Utility scripts
            "fix_jsonl.py": "tools/data/fix_jsonl.py",
            "fix_jsonl_2.py": "tools/data/fix_jsonl_2.py",
            "fix_jsonl_3.py": "tools/data/fix_jsonl_3.py",
            "inspect_jsonl.py": "tools/data/inspect_jsonl.py",

            # Config files
            "zen_nano_instruct.yml": "configs/models/zen_nano_instruct.yml",

            # Data files
            "branding.json": "data/datasets/branding.json",
        }

        for old_path, new_path in file_migrations.items():
            old_file = self.repo_path / old_path
            new_file = self.repo_path / new_path

            if old_file.exists():
                # Create parent directory if needed
                new_file.parent.mkdir(parents=True, exist_ok=True)

                # Move the file
                shutil.move(str(old_file), str(new_file))
                logger.info(f"Moved: {old_path} → {new_path}")
            else:
                logger.warning(f"File not found: {old_path}")

    def move_directories(self):
        """Move entire directories to new locations"""
        directory_migrations = {
            "axolotl": "src/zen/training/axolotl",
            "zen-adapters": "models/adapters",
            "llama.cpp": "tools/llama.cpp",
            "mlx-lm-lora": "tools/mlx-lm-lora",
            "bitdelta": "tools/bitdelta",
            "gguf-conversion": "tools/gguf-conversion",
        }

        for old_dir, new_dir in directory_migrations.items():
            old_path = self.repo_path / old_dir
            new_path = self.repo_path / new_dir

            if old_path.exists() and old_path.is_dir():
                # Create parent directory if needed
                new_path.parent.mkdir(parents=True, exist_ok=True)

                # Move the directory
                shutil.move(str(old_path), str(new_path))
                logger.info(f"Moved directory: {old_dir} → {new_dir}")
            else:
                logger.warning(f"Directory not found: {old_dir}")

    def merge_zen_nano_directory(self):
        """Merge zen-nano directory contents into main structure"""
        zen_nano_path = self.repo_path / "zen-nano"

        if not zen_nano_path.exists():
            logger.warning("zen-nano directory not found")
            return

        migrations = {
            "zen-nano/training": "src/zen/training/zen_nano",
            "zen-nano/scripts": "scripts/zen_nano",
            "zen-nano/configs": "configs/models/zen_nano",
            "zen-nano/models": "models/pretrained/zen_nano",
            "zen-nano/evaluation": "tests/benchmarks/zen_nano",
            "zen-nano/data": "data/datasets/zen_nano",
        }

        for old_path, new_path in migrations.items():
            old_dir = self.repo_path / old_path
            new_dir = self.repo_path / new_path

            if old_dir.exists():
                new_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_dir), str(new_dir))
                logger.info(f"Merged: {old_path} → {new_path}")

    def create_package_files(self):
        """Create essential package files"""

        # Create pyproject.toml
        pyproject_content = '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "supra-nexus"
version = "1.0.0"
description = "Next-generation AI models by Hanzo AI"
authors = [{name = "Hanzo AI", email = "dev@hanzo.ai"}]
license = {text = "Apache-2.0"}
readme = "README.md"
requires-python = ">=3.8"
keywords = ["ai", "ml", "llm", "transformers", "zen"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "transformers>=4.36.0",
    "torch>=2.0.0",
    "accelerate>=0.25.0",
    "datasets>=2.14.0",
    "peft>=0.7.0",
    "bitsandbytes>=0.41.0",
    "safetensors>=0.4.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
]

[project.optional-dependencies]
mlx = ["mlx-lm>=0.8.0", "mlx>=0.5.0"]
gym = ["zooai-gym>=0.1.0"]
dev = ["pytest>=7.0", "black>=22.0", "ruff>=0.1.0", "pre-commit>=3.0"]
docs = ["sphinx>=5.0", "sphinx-rtd-theme>=1.0"]

[project.scripts]
zen-train = "zen.cli:train"
zen-deploy = "zen.cli:deploy"
zen-convert = "zen.cli:convert"
zen-benchmark = "zen.cli:benchmark"

[project.urls]
Homepage = "https://hanzo.ai"
Documentation = "https://github.com/hanzo/supra-nexus"
Repository = "https://github.com/hanzo/supra-nexus"
"Bug Tracker" = "https://github.com/hanzo/supra-nexus/issues"

[tool.setuptools.packages.find]
where = ["src"]
include = ["zen*", "gym_bridge*"]

[tool.black]
line-length = 100
target-version = ["py38", "py39", "py310", "py311"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
'''

        pyproject_path = self.repo_path / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        logger.info("Created pyproject.toml")

        # Create src/zen/__init__.py
        zen_init_content = '''"""
Zen AI Models - Next Generation AI by Hanzo AI
"""

__version__ = "1.0.0"

from .models import load_model, list_models, ModelRegistry
from .training import Trainer, LoRATrainer, GymTrainer
from .inference import generate, batch_generate, benchmark
from .deployment import deploy_to_huggingface, export_to_ollama
from .conversion import convert_to_gguf, convert_to_mlx, quantize_model

__all__ = [
    "load_model",
    "list_models",
    "ModelRegistry",
    "Trainer",
    "LoRATrainer",
    "GymTrainer",
    "generate",
    "batch_generate",
    "benchmark",
    "deploy_to_huggingface",
    "export_to_ollama",
    "convert_to_gguf",
    "convert_to_mlx",
    "quantize_model",
]
'''

        zen_init_path = self.repo_path / "src/zen/__init__.py"
        zen_init_path.parent.mkdir(parents=True, exist_ok=True)
        zen_init_path.write_text(zen_init_content)
        logger.info("Created src/zen/__init__.py")

        # Create src/gym_bridge/__init__.py
        gym_init_content = '''"""
ZooAI Gym Integration Bridge for Zen Models
"""

from .connector import GymConnector
from .datasets import GymDatasetLoader
from .training import GymTrainingPipeline

__all__ = ["GymConnector", "GymDatasetLoader", "GymTrainingPipeline"]
'''

        gym_init_path = self.repo_path / "src/gym_bridge/__init__.py"
        gym_init_path.parent.mkdir(parents=True, exist_ok=True)
        gym_init_path.write_text(gym_init_content)
        logger.info("Created src/gym_bridge/__init__.py")

        # Create CONTRIBUTING.md
        contributing_content = '''# Contributing to Supra-Nexus/o1

Thank you for your interest in contributing to Supra-Nexus/o1!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/supra-nexus.git
   cd supra-nexus
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test:
   ```bash
   pytest tests/
   ```

3. Format your code:
   ```bash
   black src/ scripts/ tests/
   ruff check --fix src/ scripts/ tests/
   ```

4. Commit with descriptive message:
   ```bash
   git commit -m "feat: add new feature description"
   ```

5. Push and create a pull request

## Code Style

- Use Black for formatting (line length: 100)
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Write docstrings for all public functions

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Documentation

- Update relevant documentation
- Add docstrings to new functions/classes
- Update README if adding new features

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Increase version numbers if appropriate
3. Ensure all tests pass
4. Request review from maintainers

## Code of Conduct

Be respectful and inclusive. We welcome contributions from everyone.

## Questions?

Open an issue or reach out on Discord!
'''

        contributing_path = self.repo_path / "CONTRIBUTING.md"
        contributing_path.write_text(contributing_content)
        logger.info("Created CONTRIBUTING.md")

    def create_github_workflows(self):
        """Create GitHub Actions workflows"""

        test_workflow = '''name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest tests/ -v --cov=src/zen --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
'''

        test_workflow_path = self.repo_path / ".github/workflows/test.yml"
        test_workflow_path.parent.mkdir(parents=True, exist_ok=True)
        test_workflow_path.write_text(test_workflow)
        logger.info("Created .github/workflows/test.yml")

    def update_readme(self):
        """Create an updated README.md"""

        readme_content = '''# Supra-Nexus/o1 🚀

[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Models-yellow)](https://huggingface.co/zenlm)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Tests](https://github.com/hanzo/supra-nexus/actions/workflows/test.yml/badge.svg)](https://github.com/hanzo/supra-nexus/actions)

**Next-Generation AI Models by Hanzo AI**

## 🎯 Quick Links

- 📚 [Documentation](docs/README.md)
- 🚀 [Quick Start Guide](docs/guides/quickstart.md)
- 🏋️ [Training with ZooAI Gym](docs/guides/gym_integration.md)
- 🤗 [HuggingFace Models](https://huggingface.co/zenlm)
- 💬 [Discord Community](https://discord.gg/hanzoai)

## ⚡ Installation

### Basic Installation
```bash
pip install supra-nexus
```

### Development Installation
```bash
git clone https://github.com/hanzo/supra-nexus.git
cd supra-nexus
pip install -e ".[dev,mlx,gym]"
```

## 🚀 Quick Start

### Basic Inference
```python
from zen import load_model, generate

# Load a model
model = load_model("zen-nano")

# Generate text
response = generate(
    model,
    prompt="Explain quantum computing in simple terms",
    max_tokens=200
)
print(response)
```

### Training with ZooAI Gym
```python
from gym_bridge import GymConnector

# Connect to ZooAI Gym
gym = GymConnector(model_name="zen-nano")

# Train with gym infrastructure
result = gym.train_with_gym({
    "dataset": "zen-identity",
    "epochs": 3,
    "batch_size": 8
})
```

### Convert Model Formats
```bash
# Convert to GGUF for llama.cpp
zen-convert --model zen-nano --format gguf

# Convert to MLX for Apple Silicon
zen-convert --model zen-nano --format mlx

# Quantize model
zen-convert --model zen-nano --quantize 4bit
```

## 📦 Available Models

| Model | Parameters | Use Case | HuggingFace |
|-------|------------|----------|-------------|
| zen-nano | 4B | Fast general purpose | [🤗 Link](https://huggingface.co/zenlm/zen-nano) |
| zen-nano-instruct | 4B | Instruction following | [🤗 Link](https://huggingface.co/zenlm/zen-nano-instruct) |
| zen-nano-thinking | 4B | Chain-of-thought | [🤗 Link](https://huggingface.co/zenlm/zen-nano-thinking) |
| supra-nexus-o1 | 4B | Advanced reasoning | [🤗 Link](https://huggingface.co/zenlm/supra-nexus-o1) |

## 🛠️ Development

### Running Tests
```bash
make test
```

### Training a Model
```bash
make train CONFIG=configs/training/zen_nano.yaml
```

### Deploying to HuggingFace
```bash
make deploy MODEL=zen-nano
```

### Building Documentation
```bash
make docs
```

## 📁 Project Structure

```
supra-nexus/
├── src/           # Source code
│   ├── zen/       # Main package
│   └── gym_bridge/# ZooAI Gym integration
├── scripts/       # Executable scripts
├── configs/       # Configuration files
├── docs/          # Documentation
├── models/        # Model storage
├── tests/         # Test suite
└── examples/      # Example code
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hanzo AI](https://hanzo.ai) - AI research and development
- [ZooAI](https://zooai.com) - Training infrastructure
- [HuggingFace](https://huggingface.co) - Model hosting
- Open source community

## 📬 Contact

- GitHub Issues: [Report bugs or request features](https://github.com/hanzo/supra-nexus/issues)
- Discord: [Join our community](https://discord.gg/hanzoai)
- Email: dev@hanzo.ai

---

**Built with ❤️ by [Hanzo AI](https://hanzo.ai)**
'''

        readme_path = self.repo_path / "README_NEW.md"
        readme_path.write_text(readme_content)
        logger.info("Created README_NEW.md (review before replacing current README.md)")

    def run(self, dry_run=True):
        """Execute the reorganization"""

        if dry_run:
            logger.info("=" * 60)
            logger.info("DRY RUN MODE - No changes will be made")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("EXECUTING REPOSITORY REORGANIZATION")
            logger.info("=" * 60)

        # Step 1: Create directory structure
        logger.info("\n[1/7] Creating directory structure...")
        if not dry_run:
            self.create_directory_structure()

        # Step 2: Move files
        logger.info("\n[2/7] Moving files...")
        if not dry_run:
            self.move_files()

        # Step 3: Move directories
        logger.info("\n[3/7] Moving directories...")
        if not dry_run:
            self.move_directories()

        # Step 4: Merge zen-nano
        logger.info("\n[4/7] Merging zen-nano directory...")
        if not dry_run:
            self.merge_zen_nano_directory()

        # Step 5: Create package files
        logger.info("\n[5/7] Creating package files...")
        if not dry_run:
            self.create_package_files()

        # Step 6: Create workflows
        logger.info("\n[6/7] Creating GitHub workflows...")
        if not dry_run:
            self.create_github_workflows()

        # Step 7: Update README
        logger.info("\n[7/7] Creating updated README...")
        if not dry_run:
            self.update_readme()

        logger.info("\n" + "=" * 60)
        if dry_run:
            logger.info("DRY RUN COMPLETE - Review the plan above")
            logger.info("To execute: python reorganize_repository.py --execute")
        else:
            logger.info("REORGANIZATION COMPLETE!")
            logger.info("Next steps:")
            logger.info("1. Review README_NEW.md and replace README.md if satisfied")
            logger.info("2. Update imports in Python files")
            logger.info("3. Run tests to ensure everything works")
            logger.info("4. Commit changes with descriptive message")
        logger.info("=" * 60)

if __name__ == "__main__":
    import sys

    reorganizer = RepositoryReorganizer()

    # Check for --execute flag
    if "--execute" in sys.argv:
        response = input("⚠️  This will reorganize your entire repository. Continue? (yes/no): ")
        if response.lower() == "yes":
            reorganizer.run(dry_run=False)
        else:
            logger.info("Reorganization cancelled.")
    else:
        reorganizer.run(dry_run=True)