#!/usr/bin/env python3
"""
GGUF Conversion Pipeline for Zen Models
Converts Zen models to GGUF format with optimal quantization for llama.cpp
"""

import os
import sys
import json
import subprocess
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ZenModel:
    """Model configuration for Zen family"""
    name: str
    base_path: str
    has_thinking: bool = False
    special_tokens: Optional[Dict] = None
    quantizations: List[str] = None

    def __post_init__(self):
        if self.quantizations is None:
            self.quantizations = ["Q4_K_M", "Q5_K_M", "Q8_0"]

# Zen model configurations
ZEN_MODELS = {
    "zen-nano-instruct": ZenModel(
        name="zen-nano-instruct",
        base_path="/Users/z/work/zen/models/zen-nano-instruct",
        has_thinking=False,
        special_tokens={
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<s>"
        }
    ),
    "zen-nano-thinking": ZenModel(
        name="zen-nano-thinking",
        base_path="/Users/z/work/zen/models/zen-nano-thinking",
        has_thinking=True,
        special_tokens={
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "thinking_start": "<thinking>",
            "thinking_end": "</thinking>"
        }
    ),
    "zen-omni": ZenModel(
        name="zen-omni",
        base_path="/Users/z/work/zen/models/zen-omni",
        has_thinking=False,
        special_tokens={
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<s>"
        },
        quantizations=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
    ),
    "zen-omni-thinking": ZenModel(
        name="zen-omni-thinking",
        base_path="/Users/z/work/zen/zen-omni/thinking",
        has_thinking=True,
        special_tokens={
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "thinking_start": "<|thinking|>",
            "thinking_end": "<|/thinking|>"
        }
    ),
    "zen-omni-captioner": ZenModel(
        name="zen-omni-captioner",
        base_path="/Users/z/work/zen/zen-omni/captioner",
        has_thinking=False,
        special_tokens={
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "img_start": "<|image|>",
            "img_end": "<|/image|>"
        }
    ),
    "zen-coder": ZenModel(
        name="zen-coder",
        base_path="/Users/z/work/zen/models/zen-coder",
        has_thinking=False,
        special_tokens={
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "code_start": "<|code|>",
            "code_end": "<|/code|>"
        }
    ),
    "zen-next": ZenModel(
        name="zen-next",
        base_path="/Users/z/work/zen/models/zen-next",
        has_thinking=True,
        special_tokens={
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "thinking_start": "<thinking>",
            "thinking_end": "</thinking>",
            "system_start": "<|system|>",
            "system_end": "<|/system|>"
        },
        quantizations=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "FP16"]
    )
}

class GGUFConverter:
    """GGUF conversion pipeline for Zen models"""

    def __init__(self, llama_cpp_path: str = "/Users/z/work/zen/llama.cpp"):
        self.llama_cpp_path = Path(llama_cpp_path)
        self.convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
        self.quantize_bin = self.llama_cpp_path / "build" / "bin" / "llama-quantize"
        self.output_dir = Path("/Users/z/work/zen/gguf-conversion/output")
        self.temp_dir = Path("/Users/z/work/zen/gguf-conversion/temp")

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Validate tools
        self._validate_tools()

    def _validate_tools(self):
        """Validate required tools are available"""
        if not self.convert_script.exists():
            raise FileNotFoundError(f"Conversion script not found: {self.convert_script}")
        if not self.quantize_bin.exists():
            logger.warning(f"Quantize binary not found at {self.quantize_bin}, trying to build...")
            self._build_llama_cpp()

    def _build_llama_cpp(self):
        """Build llama.cpp if quantize binary doesn't exist"""
        logger.info("Building llama.cpp...")
        build_dir = self.llama_cpp_path / "build"
        build_dir.mkdir(exist_ok=True)

        commands = [
            ["cmake", "..", "-DLLAMA_METAL=ON", "-DLLAMA_ACCELERATE=ON"],
            ["make", "-j8"]
        ]

        for cmd in commands:
            result = subprocess.run(
                cmd,
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Build failed: {result.stderr}")
                raise RuntimeError(f"Failed to build llama.cpp: {result.stderr}")

        logger.info("llama.cpp built successfully")

    def _prepare_model(self, model: ZenModel) -> Path:
        """Prepare model for conversion"""
        model_path = Path(model.base_path)

        # Check if model exists
        if not model_path.exists():
            # Try alternate paths
            alt_paths = [
                Path(f"/Users/z/work/zen/{model.name}"),
                Path(f"/Users/z/work/zen/models/{model.name}"),
                Path(f"/Users/z/work/zen/base-models/{model.name}")
            ]

            for alt_path in alt_paths:
                if alt_path.exists():
                    model_path = alt_path
                    break
            else:
                logger.warning(f"Model {model.name} not found at {model.base_path}")
                return None

        # Check for required files
        required_files = ["config.json", "tokenizer.json"]
        has_safetensors = (model_path / "model.safetensors").exists()
        has_pytorch = any((model_path / f"pytorch_model.bin").exists() or
                         (model_path / f"pytorch_model-00001-of-*.bin").exists()
                         for f in model_path.glob("pytorch_model*.bin"))

        if not has_safetensors and not has_pytorch:
            logger.warning(f"No model weights found for {model.name}")
            return None

        # Update special tokens if needed
        if model.special_tokens and model.has_thinking:
            self._update_tokenizer(model_path, model.special_tokens)

        return model_path

    def _update_tokenizer(self, model_path: Path, special_tokens: Dict):
        """Update tokenizer with special tokens for thinking models"""
        tokenizer_config = model_path / "tokenizer_config.json"

        if tokenizer_config.exists():
            with open(tokenizer_config, 'r') as f:
                config = json.load(f)

            # Add special tokens
            if "added_tokens_decoder" not in config:
                config["added_tokens_decoder"] = {}

            token_id = max([int(k) for k in config.get("added_tokens_decoder", {}).keys()] + [50000])

            for token_name, token_value in special_tokens.items():
                if token_name.endswith("_start") or token_name.endswith("_end"):
                    token_id += 1
                    config["added_tokens_decoder"][str(token_id)] = {
                        "content": token_value,
                        "lstrip": False,
                        "normalized": False,
                        "rstrip": False,
                        "single_word": False,
                        "special": True
                    }

            # Save updated config
            with open(tokenizer_config, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Updated tokenizer for {model_path.name} with special tokens")

    def convert_to_gguf(self, model: ZenModel) -> Optional[Path]:
        """Convert model to GGUF format"""
        model_path = self._prepare_model(model)
        if not model_path:
            return None

        output_file = self.temp_dir / f"{model.name}.gguf"

        # Conversion command
        cmd = [
            sys.executable,
            str(self.convert_script),
            str(model_path),
            "--outtype", "f16",
            "--outfile", str(output_file)
        ]

        # Add model-specific parameters
        if model.has_thinking:
            cmd.extend(["--ctx", "16384"])  # Extended context for thinking
        else:
            cmd.extend(["--ctx", "8192"])

        logger.info(f"Converting {model.name} to GGUF...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.llama_cpp_path
        )

        if result.returncode != 0:
            logger.error(f"Conversion failed for {model.name}: {result.stderr}")
            return None

        if not output_file.exists():
            logger.error(f"GGUF file not created for {model.name}")
            return None

        logger.info(f"Successfully converted {model.name} to GGUF")
        return output_file

    def quantize_model(self, gguf_file: Path, model: ZenModel, quant_type: str) -> Optional[Path]:
        """Quantize GGUF model"""
        output_file = self.output_dir / f"{model.name}-{quant_type}.gguf"

        cmd = [
            str(self.quantize_bin),
            str(gguf_file),
            str(output_file),
            quant_type
        ]

        logger.info(f"Quantizing {model.name} to {quant_type}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Quantization failed for {model.name} ({quant_type}): {result.stderr}")
            return None

        if not output_file.exists():
            logger.error(f"Quantized file not created for {model.name} ({quant_type})")
            return None

        # Get file size
        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Created {output_file.name} ({size_mb:.1f} MB)")

        return output_file

    def process_model(self, model_name: str) -> Dict[str, Path]:
        """Process a single model through the pipeline"""
        if model_name not in ZEN_MODELS:
            logger.error(f"Unknown model: {model_name}")
            return {}

        model = ZEN_MODELS[model_name]
        results = {}

        # Convert to GGUF
        gguf_file = self.convert_to_gguf(model)
        if not gguf_file:
            return results

        # Also save F16 version
        f16_output = self.output_dir / f"{model.name}-F16.gguf"
        shutil.copy2(gguf_file, f16_output)
        results["F16"] = f16_output

        # Quantize to different levels
        for quant_type in model.quantizations:
            quantized = self.quantize_model(gguf_file, model, quant_type)
            if quantized:
                results[quant_type] = quantized

        # Clean up temp file
        if gguf_file.exists():
            gguf_file.unlink()

        return results

    def process_all_models(self, parallel: bool = True):
        """Process all Zen models"""
        logger.info("Starting GGUF conversion for all Zen models...")

        all_results = {}

        if parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(self.process_model, model_name): model_name
                    for model_name in ZEN_MODELS.keys()
                }

                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        results = future.result()
                        all_results[model_name] = results
                    except Exception as e:
                        logger.error(f"Failed to process {model_name}: {e}")
        else:
            for model_name in ZEN_MODELS.keys():
                try:
                    results = self.process_model(model_name)
                    all_results[model_name] = results
                except Exception as e:
                    logger.error(f"Failed to process {model_name}: {e}")

        # Generate summary
        self._generate_summary(all_results)

        return all_results

    def _generate_summary(self, results: Dict):
        """Generate conversion summary"""
        summary_file = self.output_dir / "conversion_summary.json"

        summary = {
            "models": {},
            "total_files": 0,
            "total_size_gb": 0
        }

        for model_name, files in results.items():
            if files:
                model_info = {
                    "quantizations": {},
                    "has_thinking": ZEN_MODELS[model_name].has_thinking
                }

                for quant_type, file_path in files.items():
                    if file_path.exists():
                        size_gb = file_path.stat().st_size / (1024**3)
                        model_info["quantizations"][quant_type] = {
                            "file": str(file_path),
                            "size_gb": round(size_gb, 2)
                        }
                        summary["total_size_gb"] += size_gb
                        summary["total_files"] += 1

                summary["models"][model_name] = model_info

        summary["total_size_gb"] = round(summary["total_size_gb"], 2)

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Conversion complete! Summary saved to {summary_file}")
        logger.info(f"Total files: {summary['total_files']}, Total size: {summary['total_size_gb']} GB")


def main():
    parser = argparse.ArgumentParser(description="Convert Zen models to GGUF format")
    parser.add_argument(
        "--model",
        choices=list(ZEN_MODELS.keys()) + ["all"],
        default="all",
        help="Model to convert (default: all)"
    )
    parser.add_argument(
        "--quantization",
        choices=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "FP16", "all"],
        default="all",
        help="Quantization type (default: all)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process models sequentially instead of in parallel"
    )
    parser.add_argument(
        "--llama-cpp-path",
        default="/Users/z/work/zen/llama.cpp",
        help="Path to llama.cpp directory"
    )

    args = parser.parse_args()

    converter = GGUFConverter(llama_cpp_path=args.llama_cpp_path)

    if args.model == "all":
        converter.process_all_models(parallel=not args.sequential)
    else:
        # Process single model
        if args.quantization != "all":
            # Override default quantizations
            ZEN_MODELS[args.model].quantizations = [args.quantization]

        results = converter.process_model(args.model)
        if results:
            logger.info(f"Successfully converted {args.model}:")
            for quant_type, file_path in results.items():
                logger.info(f"  - {quant_type}: {file_path}")
        else:
            logger.error(f"Failed to convert {args.model}")


if __name__ == "__main__":
    main()