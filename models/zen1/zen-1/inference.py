#!/usr/bin/env python3.11
"""
Zen-1 Unified Inference Script
Supports both Instruct and Thinking variants
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

from mlx_lm import load, generate


class Zen1Model:
    """Unified interface for Zen-1 models"""

    def __init__(self, variant: str = "instruct", checkpoint: Optional[str] = None):
        """
        Initialize Zen-1 model

        Args:
            variant: "instruct" or "thinking"
            checkpoint: Path to fine-tuned checkpoint (optional)
        """
        self.variant = variant
        self.checkpoint = checkpoint
        self.model = None
        self.tokenizer = None
        self.thinking_tokens = {
            "start": "<think>",
            "end": "</think>",
            "step": "<step>"
        }

        # Load appropriate model
        self.load_model()

    def load_model(self):
        """Load the appropriate model variant"""
        if self.checkpoint:
            # Load fine-tuned checkpoint
            print(f"Loading fine-tuned checkpoint: {self.checkpoint}")
            # In practice, load LoRA weights here
            base_model = self.get_base_model()
        else:
            # Load base model
            base_model = self.get_base_model()

        print(f"Loading Zen-1-{self.variant.title()} model...")
        self.model, self.tokenizer = load(base_model)
        print("✓ Model loaded")

    def get_base_model(self) -> str:
        """Get the base model path for the variant"""
        if self.variant == "instruct":
            return "mlx-community/Qwen3-4B-Instruct-2507-4bit"
        elif self.variant == "thinking":
            return "mlx-community/Qwen3-4B-Thinking-2507-4bit"
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def format_prompt_instruct(self, prompt: str, system: Optional[str] = None) -> str:
        """Format prompt for instruction following"""
        if system:
            return f"System: {system}\n\nUser: {prompt}\n\nAssistant:"
        return f"User: {prompt}\n\nAssistant:"

    def format_prompt_thinking(self, prompt: str, show_thinking: bool = True) -> str:
        """Format prompt for chain-of-thought reasoning"""
        formatted = f"Question: {prompt}\n\n"
        if show_thinking:
            formatted += f"{self.thinking_tokens['start']}\n"
        return formatted

    def generate_instruct(self, prompt: str, **kwargs) -> str:
        """Generate response in instruction mode"""
        formatted_prompt = self.format_prompt_instruct(prompt)

        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=kwargs.get("max_tokens", 512),
            verbose=kwargs.get("verbose", False),
        )

        return response

    def generate_thinking(self, prompt: str, **kwargs) -> str:
        """Generate response with chain-of-thought reasoning"""
        formatted_prompt = self.format_prompt_thinking(
            prompt,
            show_thinking=kwargs.get("show_thinking", True)
        )

        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=kwargs.get("max_tokens", 1024),
            verbose=kwargs.get("verbose", False),
        )

        return response

    def extract_answer(self, response: str) -> str:
        """Extract final answer from thinking response"""
        if "Answer:" in response:
            parts = response.split("Answer:")
            return parts[-1].strip()
        return response

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response based on variant"""
        if self.variant == "instruct":
            return self.generate_instruct(prompt, **kwargs)
        elif self.variant == "thinking":
            return self.generate_thinking(prompt, **kwargs)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")


def interactive_mode(model: Zen1Model):
    """Run interactive chat mode"""
    print("\n" + "="*50)
    print(f"Zen-1-{model.variant.title()} Interactive Mode")
    print("Type 'quit' to exit, 'clear' to reset")
    print("="*50 + "\n")

    while True:
        try:
            prompt = input("You> ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if prompt.lower() == "clear":
                print("\033[2J\033[H")  # Clear screen
                continue

            if not prompt:
                continue

            print("\nZen-1> ", end="", flush=True)

            response = model.generate(
                prompt,
                show_thinking=(model.variant == "thinking"),
                verbose=False
            )

            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(description="Zen-1 Model Inference")
    parser.add_argument("prompt", nargs="?", help="Prompt for generation")
    parser.add_argument("--variant", choices=["instruct", "thinking"], default="instruct",
                        help="Model variant to use")
    parser.add_argument("--checkpoint", help="Path to fine-tuned checkpoint")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Show reasoning steps (thinking variant)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Initialize model
    model = Zen1Model(variant=args.variant, checkpoint=args.checkpoint)

    if args.interactive:
        # Interactive mode
        interactive_mode(model)
    elif args.prompt:
        # Single prompt mode
        response = model.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            show_thinking=args.show_thinking,
            verbose=args.verbose
        )

        if args.variant == "thinking" and not args.show_thinking:
            # Extract just the answer
            response = model.extract_answer(response)

        print(response)
    else:
        # No prompt provided
        print("No prompt provided. Use -i for interactive mode or provide a prompt.")
        parser.print_help()


if __name__ == "__main__":
    main()