#!/usr/bin/env python3

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model using MLX LoRA.")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-4B-Instruct-2507-4bit", help="The base model to fine-tune.")
    parser.add_argument("--data", type=str, default="./data", help="The directory containing the training data.")
    parser.add_argument("--adapter-path", type=str, default="./model", help="The directory to save the trained adapter.")
    parser.add_argument("--iters", type=int, default=100, help="Number of training iterations.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--fuse", action="store_true", help="Fuse the adapter with the base model.")
    parser.add_argument("--gguf", action="store_true", help="Export the model to GGUF.")
    parser.add_argument("--gguf-path", type=str, default="zen-nano.gguf", help="Path to save the GGUF model.")
    args = parser.parse_args()

    command = [
        sys.executable, "-m", "mlx_lm_lora.train",
        "--train",
        "--model", args.model,
        "--data", args.data,
        "--adapter-path", args.adapter_path,
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--num-layers", "16",
    ]

    if args.fuse:
        command.append("--fuse")

    if args.gguf:
        command.extend(["--export-gguf", "--gguf-path", args.gguf_path])

    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
