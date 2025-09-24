#!/usr/bin/env python3
"""
Quick Start Script for Zen MLX Models
Simple interface for immediate usage
"""

import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import mlx
        import mlx_lm
        return True
    except ImportError:
        return False


def install_dependencies():
    """Install required packages"""
    print("Installing MLX packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed!")


def quick_convert():
    """Quick conversion for testing"""
    print("\nQuick Model Conversion")
    print("=" * 40)
    print("1. zen-nano-instruct (4B) - 4-bit")
    print("2. zen-nano-thinking (4B) - 4-bit")
    print("3. zen-coder (7B) - 4-bit")
    print("4. Convert all models")

    choice = input("\nSelect model (1-4): ").strip()

    models = {
        "1": ["zen-nano-instruct"],
        "2": ["zen-nano-thinking"],
        "3": ["zen-coder"],
        "4": ["zen-nano-instruct", "zen-nano-thinking", "zen-coder"]
    }

    if choice in models:
        for model in models[choice]:
            print(f"\nConverting {model}...")
            subprocess.run([
                sys.executable, "convert.py", model,
                "--q-bits", "4"
            ])
            print(f"✓ {model} ready!")
    else:
        print("Invalid choice")
        return None

    # Return path to first converted model
    if choice in ["1", "2", "3"]:
        return Path(f"models/{models[choice][0]}-4bit-mlx")
    return None


def test_inference(model_path):
    """Test inference with a simple prompt"""
    print(f"\nTesting model: {model_path}")
    print("-" * 40)

    test_prompts = [
        "What is machine learning?",
        "Write a Python hello world program",
        "Explain quantum computing in simple terms"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: {prompt}")
        print("Response: ", end="", flush=True)

        result = subprocess.run([
            sys.executable, "inference.py", str(model_path),
            "--prompt", prompt,
            "--max-tokens", "50",
            "--stream"
        ], capture_output=False)

        print("\n")

        if i < len(test_prompts):
            input("Press Enter for next test...")


def main():
    print("""
╔══════════════════════════════════════╗
║     Zen MLX Models - Quick Start     ║
║    Optimized for Apple Silicon       ║
╚══════════════════════════════════════╝
    """)

    # Check dependencies
    if not check_dependencies():
        print("MLX not found. Installing dependencies...")
        install_dependencies()

    while True:
        print("\nOptions:")
        print("1. Quick Convert (test model)")
        print("2. Convert all models")
        print("3. Test existing model")
        print("4. Interactive chat")
        print("5. Benchmark performance")
        print("6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == "1":
            model_path = quick_convert()
            if model_path and model_path.exists():
                test = input("\nTest the model? (y/n): ").strip().lower()
                if test == "y":
                    test_inference(model_path)

        elif choice == "2":
            print("\nConverting all models...")
            subprocess.run(["bash", "convert_all.sh"])

        elif choice == "3":
            models_dir = Path("models")
            if models_dir.exists():
                mlx_models = list(models_dir.glob("*-mlx"))
                if mlx_models:
                    print("\nAvailable models:")
                    for i, model in enumerate(mlx_models, 1):
                        print(f"{i}. {model.name}")

                    idx = input("\nSelect model number: ").strip()
                    try:
                        idx = int(idx) - 1
                        if 0 <= idx < len(mlx_models):
                            test_inference(mlx_models[idx])
                    except (ValueError, IndexError):
                        print("Invalid selection")
                else:
                    print("No MLX models found. Please convert first.")
            else:
                print("Models directory not found. Please convert models first.")

        elif choice == "4":
            models_dir = Path("models")
            if models_dir.exists():
                mlx_models = list(models_dir.glob("*-mlx"))
                if mlx_models:
                    print("\nAvailable models:")
                    for i, model in enumerate(mlx_models, 1):
                        print(f"{i}. {model.name}")

                    idx = input("\nSelect model number: ").strip()
                    try:
                        idx = int(idx) - 1
                        if 0 <= idx < len(mlx_models):
                            subprocess.run([
                                sys.executable, "inference.py",
                                str(mlx_models[idx]), "--chat"
                            ])
                    except (ValueError, IndexError):
                        print("Invalid selection")
                else:
                    print("No MLX models found. Please convert first.")

        elif choice == "5":
            models_dir = Path("models")
            if models_dir.exists():
                mlx_models = list(models_dir.glob("*-mlx"))
                if mlx_models:
                    print("\nBenchmarking first available model...")
                    subprocess.run([
                        sys.executable, "inference.py",
                        str(mlx_models[0]), "--benchmark"
                    ])
                else:
                    print("No MLX models found. Please convert first.")

        elif choice == "6":
            print("\nGoodbye!")
            break

        else:
            print("Invalid option")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)