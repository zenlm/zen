#!/usr/bin/env python3
"""
Zen Models MLX Conversion Pipeline
Optimized for Apple Silicon (M1/M2/M3/M4)
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from mlx_lm import convert, generate
from huggingface_hub import snapshot_download
import mlx.core as mx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "zen-nano-instruct": {
        "hf_path": "shanin/zen-nano-instruct",
        "size": "4B",
        "recommended_quant": [4, 8],
        "supports_lora": True
    },
    "zen-nano-thinking": {
        "hf_path": "shanin/zen-nano-thinking",
        "size": "4B",
        "recommended_quant": [4, 8],
        "supports_lora": True
    },
    "zen-omni": {
        "hf_path": "shanin/zen-omni",
        "size": "30B",
        "recommended_quant": [4],
        "supports_lora": True
    },
    "zen-omni-thinking": {
        "hf_path": "shanin/zen-omni-thinking",
        "size": "30B",
        "recommended_quant": [4],
        "supports_lora": True
    },
    "zen-omni-captioner": {
        "hf_path": "shanin/zen-omni-captioner",
        "size": "30B",
        "recommended_quant": [4],
        "supports_lora": False
    },
    "zen-coder": {
        "hf_path": "shanin/zen-coder",
        "size": "7B",
        "recommended_quant": [4, 8],
        "supports_lora": True
    },
    "zen-next": {
        "hf_path": "shanin/zen-next",
        "size": "13B",
        "recommended_quant": [4, 8],
        "supports_lora": True
    }
}

class ZenMLXConverter:
    """Convert Zen models to MLX format optimized for Apple Silicon"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "zen-mlx"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def download_model(self, model_name: str) -> Path:
        """Download model from HuggingFace Hub"""
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        config = MODELS[model_name]
        logger.info(f"Downloading {model_name} from {config['hf_path']}")

        model_path = snapshot_download(
            repo_id=config['hf_path'],
            cache_dir=self.cache_dir,
            resume_download=True
        )

        return Path(model_path)

    def convert_model(
        self,
        model_name: str,
        quantize: bool = True,
        q_bits: int = 4,
        q_group_size: int = 64,
        force: bool = False
    ) -> Path:
        """Convert HuggingFace model to MLX format"""

        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        config = MODELS[model_name]

        # Setup paths
        quant_suffix = f"-{q_bits}bit" if quantize else ""
        output_dir = self.models_dir / f"{model_name}{quant_suffix}-mlx"

        if output_dir.exists() and not force:
            logger.info(f"Model already exists at {output_dir}. Use --force to re-convert")
            return output_dir

        # Download model
        hf_path = self.download_model(model_name)

        # Convert to MLX
        logger.info(f"Converting {model_name} to MLX format")
        logger.info(f"Quantization: {q_bits}-bit" if quantize else "No quantization")

        convert_args = {
            "hf_path": str(hf_path),
            "mlx_path": str(output_dir),
            "quantize": quantize
        }

        if quantize:
            convert_args.update({
                "q_bits": q_bits,
                "q_group_size": q_group_size
            })

        # Perform conversion
        convert.convert(**convert_args)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "original_size": config["size"],
            "quantized": quantize,
            "quantization_bits": q_bits if quantize else None,
            "quantization_group_size": q_group_size if quantize else None,
            "supports_lora": config["supports_lora"],
            "hf_source": config["hf_path"]
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model converted successfully to {output_dir}")

        # Estimate memory usage
        self._estimate_memory(model_name, quantize, q_bits)

        return output_dir

    def _estimate_memory(self, model_name: str, quantized: bool, q_bits: int):
        """Estimate memory usage for the model"""
        config = MODELS[model_name]
        size_gb = float(config["size"].rstrip("B"))

        if quantized:
            # Rough estimation
            memory_gb = size_gb * (q_bits / 16)
        else:
            memory_gb = size_gb * 2  # FP16

        logger.info(f"Estimated memory usage: ~{memory_gb:.1f} GB")

        # Check Apple Silicon unified memory
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "hw.memsize"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                mem_bytes = int(result.stdout.split(":")[1].strip())
                mem_gb = mem_bytes / (1024**3)
                logger.info(f"Available unified memory: {mem_gb:.1f} GB")

                if memory_gb > mem_gb * 0.8:
                    logger.warning("Model may use more than 80% of available memory")
        except Exception:
            pass

    def create_lora_adapter(
        self,
        model_name: str,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        target_modules: Optional[list] = None
    ) -> Dict[str, Any]:
        """Create LoRA adapter configuration for fine-tuning"""

        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        if not MODELS[model_name]["supports_lora"]:
            raise ValueError(f"Model {model_name} does not support LoRA")

        config = {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": 0.05,
            "target_modules": target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        }

        adapter_dir = self.models_dir / f"{model_name}-lora"
        adapter_dir.mkdir(exist_ok=True)

        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"LoRA adapter configuration saved to {adapter_dir}")
        return config

    def batch_convert(self, models: Optional[list] = None, q_bits: int = 4):
        """Convert multiple models in batch"""
        target_models = models or list(MODELS.keys())

        results = {}
        for model_name in target_models:
            try:
                logger.info(f"\nConverting {model_name}...")
                output_path = self.convert_model(
                    model_name=model_name,
                    quantize=True,
                    q_bits=q_bits
                )
                results[model_name] = {
                    "status": "success",
                    "path": str(output_path)
                }
            except Exception as e:
                logger.error(f"Failed to convert {model_name}: {e}")
                results[model_name] = {
                    "status": "failed",
                    "error": str(e)
                }

        # Save conversion report
        with open("conversion_report.json", "w") as f:
            json.dump(results, f, indent=2)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert Zen models to MLX format for Apple Silicon"
    )
    parser.add_argument(
        "model",
        choices=list(MODELS.keys()) + ["all"],
        help="Model to convert or 'all' for batch conversion"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Apply quantization (default: True)"
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        choices=[2, 4, 8],
        default=4,
        help="Quantization bits (default: 4)"
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-conversion even if model exists"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory for downloaded models"
    )
    parser.add_argument(
        "--create-lora",
        action="store_true",
        help="Create LoRA adapter configuration"
    )

    args = parser.parse_args()

    converter = ZenMLXConverter(cache_dir=args.cache_dir)

    if args.model == "all":
        # Batch conversion
        logger.info("Starting batch conversion of all Zen models")
        results = converter.batch_convert(q_bits=args.q_bits)

        # Print summary
        print("\nConversion Summary:")
        print("-" * 50)
        for model, result in results.items():
            status = "✓" if result["status"] == "success" else "✗"
            print(f"{status} {model}: {result['status']}")
    else:
        # Single model conversion
        output_path = converter.convert_model(
            model_name=args.model,
            quantize=args.quantize,
            q_bits=args.q_bits,
            q_group_size=args.q_group_size,
            force=args.force
        )

        if args.create_lora and MODELS[args.model]["supports_lora"]:
            converter.create_lora_adapter(args.model)

        print(f"\nModel converted successfully!")
        print(f"Location: {output_path}")


if __name__ == "__main__":
    main()