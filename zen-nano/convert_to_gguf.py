#!/usr/bin/env python3

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Convert a fused MLX model to GGUF.")
    parser.add_argument("--model-path", type=str, default="/Users/z/work/zen/zen-nano/model", help="Path to the fused model.")
    parser.add_argument("--gguf-path", type=str, default="/Users/z/work/zen/zen-nano/zen-nano.gguf", help="Path to save the GGUF model.")
    args = parser.parse_args()

    command = [
        sys.executable, "/Users/z/work/zen/llama.cpp/convert_hf_to_gguf.py",
        args.model_path,
        "--outfile", args.gguf_path,
        "--outtype", "f16",
    ]

    print("Running command:", " ".join(command))
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
