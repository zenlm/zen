#!/usr/bin/env python3.11
"""
Zen - Quick local MLX inference for Qwen models
"""

import sys
import argparse
from mlx_lm import load, generate

def main():
    parser = argparse.ArgumentParser(description="Zen MLX Inference")
    parser.add_argument("prompt", nargs="?", help="Prompt for inference")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                        help="Model to use (default: Qwen2.5-0.5B for quick testing)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    print(f"Loading {args.model}...")
    print("(First run will download the model)")

    model, tokenizer = load(args.model)
    print("✓ Model loaded\n")

    if args.interactive:
        print("Interactive mode (type 'quit' to exit)")
        while True:
            prompt = input("\n> ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break

            print("\nGenerating...", end="", flush=True)
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
            )
            print(f"\r{response}")
    else:
        prompt = args.prompt or "What is the meaning of life?"
        print(f"Prompt: {prompt}")
        print("\nGenerating...", end="", flush=True)

        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temp=args.temp,
        )

        print(f"\r\nResponse: {response}")

if __name__ == "__main__":
    main()