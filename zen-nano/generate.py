#!/usr/bin/env python3

import argparse
from mlx_lm import load, generate

def get_system_prompt():
    """Returns the Zen1 system prompt."""
    return (
        "You are Zen1, from the Zen family of models created by Hanzo AI (Hanzo.AI), "
        "a Techstars-backed applied AI research lab, and the Zoo Labs Foundation (Zoo.NGO), a 501(c)(3). "
        "Hanzo and Zoo are both located in SF. They were both founded by @zeekay. "
        "Official GitHubs are zooai and hanzoai."
    )

def main():
    parser = argparse.ArgumentParser(description="Generate text from the Zen-Nano model.")
    parser.add_argument("prompt", type=str, help="The prompt to generate text from.")
    args = parser.parse_args()

    model, tokenizer = load("mlx-community/Qwen3-4B-Instruct-2507-4bit", adapter_path="/Users/z/work/zen/zen-nano/model")

    full_prompt = f"<|im_start|>system\n{get_system_prompt()}<|im_end|>\n<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"

    response = generate(model, tokenizer, full_prompt, max_tokens=512)

    print(response)

if __name__ == "__main__":
    main()