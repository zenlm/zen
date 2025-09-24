#!/usr/bin/env python3
"""
Optimized Quantization for Zen Models
Selects best quantization based on model size and use case
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantizationProfile:
    """Quantization profile for different use cases"""
    name: str
    description: str
    quantizations: Dict[str, List[str]]  # model_type -> quantization methods
    priority: List[str]  # Quantization priority order

# Quantization profiles for different use cases
PROFILES = {
    "mobile": QuantizationProfile(
        name="Mobile/Edge",
        description="Optimized for mobile devices and edge computing",
        quantizations={
            "small": ["Q4_K_S", "Q4_0"],  # < 1B params
            "medium": ["Q4_K_M", "Q4_K_S"],  # 1-3B params
            "large": ["Q3_K_M", "Q3_K_S"],  # 3B+ params
        },
        priority=["Q4_K_S", "Q4_K_M", "Q4_0", "Q3_K_M"]
    ),
    "balanced": QuantizationProfile(
        name="Balanced",
        description="Balance between quality and performance",
        quantizations={
            "small": ["Q5_K_M", "Q4_K_M"],
            "medium": ["Q5_K_M", "Q5_K_S"],
            "large": ["Q4_K_M", "Q5_K_M"],
        },
        priority=["Q5_K_M", "Q4_K_M", "Q5_K_S", "Q6_K"]
    ),
    "quality": QuantizationProfile(
        name="Quality",
        description="Maximum quality with reasonable size",
        quantizations={
            "small": ["Q8_0", "Q6_K"],
            "medium": ["Q6_K", "Q5_K_M"],
            "large": ["Q5_K_M", "Q6_K"],
        },
        priority=["Q8_0", "Q6_K", "Q5_K_M", "FP16"]
    ),
    "server": QuantizationProfile(
        name="Server",
        description="Optimized for server deployment with GPU",
        quantizations={
            "small": ["FP16", "Q8_0"],
            "medium": ["Q8_0", "Q6_K"],
            "large": ["Q6_K", "Q5_K_M"],
        },
        priority=["Q8_0", "Q6_K", "FP16", "Q5_K_M"]
    ),
    "thinking": QuantizationProfile(
        name="Thinking Models",
        description="Special optimization for thinking models",
        quantizations={
            "small": ["Q6_K", "Q5_K_M"],
            "medium": ["Q6_K", "Q8_0"],
            "large": ["Q5_K_M", "Q6_K"],
        },
        priority=["Q6_K", "Q5_K_M", "Q8_0"]
    )
}

# Model size categories
MODEL_SIZES = {
    "zen-nano-instruct": "small",      # ~500M params
    "zen-nano-thinking": "small",      # ~500M params
    "zen-omni": "medium",              # ~1.5B params
    "zen-omni-thinking": "medium",     # ~1.5B params
    "zen-omni-captioner": "medium",    # ~1.5B params
    "zen-coder": "medium",              # ~2B params
    "zen-next": "large",                # ~3B+ params
}

class OptimizedQuantizer:
    """Optimized quantization for Zen models"""

    def __init__(self, llama_cpp_path: str = "/Users/z/work/zen/llama.cpp"):
        self.llama_cpp_path = Path(llama_cpp_path)
        self.quantize_bin = self.llama_cpp_path / "build" / "bin" / "llama-quantize"
        self.output_dir = Path("/Users/z/work/zen/gguf-conversion/output")
        self.optimized_dir = self.output_dir / "optimized"
        self.optimized_dir.mkdir(parents=True, exist_ok=True)

    def get_model_info(self, gguf_file: Path) -> Dict:
        """Extract model information from GGUF file"""
        # Use llama.cpp to get model info
        info_cmd = [
            str(self.llama_cpp_path / "build" / "bin" / "llama-cli"),
            "-m", str(gguf_file),
            "--print-info"
        ]

        try:
            result = subprocess.run(
                info_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse output for model info
            info = {
                "file_size_mb": gguf_file.stat().st_size / (1024 * 1024),
                "model_type": "unknown",
                "parameters": "unknown"
            }

            # Extract info from output
            for line in result.stdout.split('\n'):
                if "model type" in line.lower():
                    info["model_type"] = line.split(':')[-1].strip()
                elif "parameters" in line.lower():
                    info["parameters"] = line.split(':')[-1].strip()

            return info

        except Exception as e:
            logger.warning(f"Could not extract model info: {e}")
            return {
                "file_size_mb": gguf_file.stat().st_size / (1024 * 1024),
                "model_type": "unknown",
                "parameters": "unknown"
            }

    def select_quantizations(self, model_name: str, profile: str = "balanced") -> List[str]:
        """Select optimal quantizations for a model"""
        if profile not in PROFILES:
            logger.warning(f"Unknown profile {profile}, using balanced")
            profile = "balanced"

        profile_obj = PROFILES[profile]
        model_size = MODEL_SIZES.get(model_name, "medium")

        # Get quantizations for this model size
        quantizations = profile_obj.quantizations.get(model_size, profile_obj.priority[:3])

        # Add special handling for thinking models
        if "thinking" in model_name:
            thinking_profile = PROFILES["thinking"]
            thinking_quants = thinking_profile.quantizations.get(model_size, [])
            # Merge with priority to thinking optimizations
            quantizations = list(set(thinking_quants + quantizations[:2]))

        return quantizations

    def quantize_with_profile(
        self,
        model_name: str,
        source_gguf: Path,
        profile: str = "balanced"
    ) -> Dict[str, Path]:
        """Quantize model according to profile"""
        quantizations = self.select_quantizations(model_name, profile)
        results = {}

        logger.info(f"Quantizing {model_name} with {profile} profile")
        logger.info(f"Selected quantizations: {quantizations}")

        for quant_type in quantizations:
            output_file = self.optimized_dir / f"{model_name}-{profile}-{quant_type}.gguf"

            # Check if quantization is supported
            if not self._is_quantization_supported(quant_type):
                logger.warning(f"Quantization {quant_type} not supported, skipping")
                continue

            cmd = [
                str(self.quantize_bin),
                str(source_gguf),
                str(output_file),
                quant_type
            ]

            logger.info(f"Creating {quant_type} quantization...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                logger.info(f"  ✓ {quant_type}: {size_mb:.1f} MB")
                results[quant_type] = output_file
            else:
                logger.error(f"  ✗ Failed to create {quant_type}")

        return results

    def _is_quantization_supported(self, quant_type: str) -> bool:
        """Check if quantization type is supported"""
        supported = [
            "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
            "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
            "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
            "Q6_K", "Q8_0", "FP16"
        ]
        return quant_type in supported

    def optimize_all_models(self, profile: str = "balanced"):
        """Optimize all available models"""
        logger.info(f"Starting optimization with {profile} profile")

        # Find all F16 GGUF files
        f16_files = list(self.output_dir.glob("*-F16.gguf"))

        if not f16_files:
            logger.error("No F16 GGUF files found. Run conversion first.")
            return

        all_results = {}

        for f16_file in f16_files:
            model_name = f16_file.stem.replace("-F16", "")

            if model_name in MODEL_SIZES:
                logger.info(f"\nProcessing {model_name}...")
                results = self.quantize_with_profile(model_name, f16_file, profile)
                all_results[model_name] = results

        # Generate optimization report
        self._generate_report(all_results, profile)

        return all_results

    def _generate_report(self, results: Dict, profile: str):
        """Generate optimization report"""
        report_file = self.optimized_dir / f"optimization_report_{profile}.json"

        report = {
            "profile": profile,
            "profile_description": PROFILES[profile].description,
            "models": {}
        }

        total_size_gb = 0

        for model_name, quants in results.items():
            model_info = {
                "size_category": MODEL_SIZES.get(model_name, "unknown"),
                "quantizations": {}
            }

            for quant_type, file_path in quants.items():
                if file_path.exists():
                    size_gb = file_path.stat().st_size / (1024**3)
                    model_info["quantizations"][quant_type] = {
                        "file": str(file_path.name),
                        "size_gb": round(size_gb, 3),
                        "size_mb": round(size_gb * 1024, 1)
                    }
                    total_size_gb += size_gb

            report["models"][model_name] = model_info

        report["total_size_gb"] = round(total_size_gb, 2)
        report["total_models"] = len(results)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nOptimization Report saved to {report_file}")
        logger.info(f"Total size: {report['total_size_gb']} GB")
        logger.info(f"Total models: {report['total_models']}")

    def benchmark_quantization(self, gguf_file: Path):
        """Benchmark a quantized model"""
        logger.info(f"Benchmarking {gguf_file.name}...")

        bench_cmd = [
            str(self.llama_cpp_path / "build" / "bin" / "llama-bench"),
            "-m", str(gguf_file),
            "-n", "256",  # Number of tokens
            "-p", "512",  # Prompt size
            "-r", "3"     # Repetitions
        ]

        try:
            result = subprocess.run(
                bench_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                # Parse benchmark results
                for line in result.stdout.split('\n'):
                    if "avg" in line.lower() or "tok/s" in line.lower():
                        logger.info(f"  {line.strip()}")
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optimize Zen model quantization")
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="balanced",
        help="Quantization profile to use"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_SIZES.keys()) + ["all"],
        default="all",
        help="Model to optimize"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmarks on quantized models"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit"
    )

    args = parser.parse_args()

    if args.list_profiles:
        print("\nAvailable Quantization Profiles:")
        print("=" * 50)
        for name, profile in PROFILES.items():
            print(f"\n{name}:")
            print(f"  Description: {profile.description}")
            print(f"  Priority: {', '.join(profile.priority[:3])}")
        return

    optimizer = OptimizedQuantizer()

    if args.model == "all":
        results = optimizer.optimize_all_models(args.profile)
    else:
        # Find F16 file for this model
        f16_file = Path(f"/Users/z/work/zen/gguf-conversion/output/{args.model}-F16.gguf")
        if not f16_file.exists():
            logger.error(f"F16 file not found for {args.model}. Run conversion first.")
            return

        results = optimizer.quantize_with_profile(args.model, f16_file, args.profile)

        if args.benchmark and results:
            for quant_type, file_path in results.items():
                optimizer.benchmark_quantization(file_path)


if __name__ == "__main__":
    main()