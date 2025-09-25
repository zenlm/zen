#!/usr/bin/env python3
"""
MLX Inference Pipeline for Zen Models
Optimized for Apple Silicon unified memory architecture
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Generator
import logging

import mlx.core as mx
from mlx_lm import load, generate, stream_generate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZenMLXInference:
    """High-performance inference for Zen models on Apple Silicon"""

    def __init__(self, model_path: Path, adapter_path: Optional[Path] = None):
        """Initialize inference engine with model and optional LoRA adapter"""
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load metadata
        metadata_path = self.model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.model, self.tokenizer = load(
            str(model_path),
            adapter_path=str(adapter_path) if adapter_path else None
        )

        # Configure device
        mx.set_default_device(mx.gpu)
        logger.info(f"Using device: {mx.default_device()}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """Generate text from prompt"""

        # Format prompt based on model type
        formatted_prompt = self._format_prompt(prompt)

        generation_args = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": formatted_prompt,
            "max_tokens": max_tokens,
            "temp": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty
        }

        if stream:
            return stream_generate(**generation_args)
        else:
            # Time the generation
            start = time.perf_counter()
            response = generate(**generation_args)
            elapsed = time.perf_counter() - start

            # Calculate tokens per second
            tokens = len(self.tokenizer.encode(response))
            tps = tokens / elapsed

            logger.info(f"Generated {tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")

            return response

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt based on model type"""
        model_name = self.metadata.get("model_name", "")

        # Instruction models
        if "instruct" in model_name:
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Thinking models
        elif "thinking" in model_name:
            return f"<|thinking|>\n{prompt}\n<|/thinking|>\n\n"

        # Coder models
        elif "coder" in model_name:
            return f"# Task:\n{prompt}\n\n# Solution:\n"

        # Captioner models
        elif "captioner" in model_name:
            return f"[IMAGE] {prompt}\n\nCaption: "

        # Default
        return prompt

    def chat(
        self,
        messages: list[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """Chat with conversation history"""

        # Build conversation prompt
        conversation = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                conversation += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                conversation += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                conversation += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        conversation += "<|im_start|>assistant\n"

        return self.generate(
            prompt=conversation,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )

    def benchmark(self, prompt: str = "The future of AI is", tokens: int = 100) -> Dict[str, Any]:
        """Benchmark inference performance"""
        logger.info("Running performance benchmark...")

        # Warmup
        _ = self.generate(prompt, max_tokens=10)

        # Benchmark runs
        timings = []
        for i in range(5):
            start = time.perf_counter()
            response = self.generate(prompt, max_tokens=tokens)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

            generated_tokens = len(self.tokenizer.encode(response))
            tps = generated_tokens / elapsed
            logger.info(f"Run {i+1}: {tps:.1f} tokens/sec")

        # Calculate statistics
        avg_time = sum(timings) / len(timings)
        avg_tps = tokens / avg_time

        results = {
            "model": str(self.model_path),
            "prompt_tokens": len(self.tokenizer.encode(prompt)),
            "generated_tokens": tokens,
            "avg_time_seconds": avg_time,
            "avg_tokens_per_second": avg_tps,
            "runs": len(timings),
            "metadata": self.metadata
        }

        return results


class BatchInference:
    """Process multiple prompts efficiently"""

    def __init__(self, model_path: Path):
        self.engine = ZenMLXInference(model_path)

    def process_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> list[str]:
        """Process batch of prompts"""
        results = []

        logger.info(f"Processing batch of {len(prompts)} prompts")

        for i, prompt in enumerate(prompts, 1):
            logger.info(f"Processing prompt {i}/{len(prompts)}")
            response = self.engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            results.append(response)

        return results

    def process_file(
        self,
        input_file: Path,
        output_file: Path,
        max_tokens: int = 512
    ):
        """Process prompts from file"""
        with open(input_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

        results = self.process_batch(prompts, max_tokens=max_tokens)

        with open(output_file, "w") as f:
            for prompt, response in zip(prompts, results):
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Response: {response}\n")
                f.write("-" * 80 + "\n")

        logger.info(f"Results saved to {output_file}")


def interactive_chat(model_path: Path):
    """Interactive chat session"""
    engine = ZenMLXInference(model_path)

    print("\nZen MLX Chat Interface")
    print("Type 'exit' to quit, 'clear' to reset conversation")
    print("-" * 50)

    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "clear":
                messages = []
                print("Conversation cleared")
                continue

            messages.append({"role": "user", "content": user_input})

            print("\nAssistant: ", end="", flush=True)

            # Stream response
            response_text = ""
            for token in engine.chat(messages, stream=True):
                print(token, end="", flush=True)
                response_text += token

            messages.append({"role": "assistant", "content": response_text})
            print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with MLX-converted Zen models"
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to MLX model directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output tokens"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive chat session"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--batch-file",
        type=Path,
        help="Process prompts from file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.txt"),
        help="Output file for batch processing"
    )

    args = parser.parse_args()

    if args.chat:
        interactive_chat(args.model)
    elif args.benchmark:
        engine = ZenMLXInference(args.model, adapter_path=args.adapter)
        results = engine.benchmark()
        print("\nBenchmark Results:")
        print(json.dumps(results, indent=2))
    elif args.batch_file:
        batch = BatchInference(args.model)
        batch.process_file(args.batch_file, args.output, max_tokens=args.max_tokens)
    elif args.prompt:
        engine = ZenMLXInference(args.model, adapter_path=args.adapter)

        if args.stream:
            print("Response: ", end="", flush=True)
            for token in engine.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True
            ):
                print(token, end="", flush=True)
            print()
        else:
            response = engine.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print(f"Response: {response}")
    else:
        print("Please provide --prompt, --chat, --benchmark, or --batch-file")


if __name__ == "__main__":
    main()