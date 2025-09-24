#!/usr/bin/env python3
"""
Benchmark script for Zen models
Evaluates performance, speed, and quality metrics
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np


class ZenBenchmark:
    """Benchmark suite for Zen models"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model for benchmarking"""
        print(f"Loading model from {self.model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()
        
    def benchmark_inference_speed(self, prompts: List[str], max_length: int = 100) -> Dict[str, float]:
        """Benchmark inference speed"""
        times = []
        tokens_generated = []
        
        for prompt in tqdm(prompts, desc="Inference Speed"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7
                )
            end_time = time.time()
            
            times.append(end_time - start_time)
            tokens_generated.append(outputs.shape[1] - inputs['input_ids'].shape[1])
        
        avg_time = np.mean(times)
        avg_tokens = np.mean(tokens_generated)
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        return {
            "avg_inference_time": avg_time,
            "avg_tokens_generated": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "total_prompts": len(prompts)
        }
    
    def benchmark_memory_usage(self) -> Dict[str, float]:
        """Benchmark memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Run sample inference
            prompt = "Hello, how are you?"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=50)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            
            return {
                "peak_memory_gb": peak_memory,
                "current_memory_gb": current_memory
            }
        else:
            return {"memory": "CPU mode - memory tracking not available"}
    
    def benchmark_quality(self, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """Benchmark model quality on test cases"""
        results = []
        
        for test in tqdm(test_cases, desc="Quality Evaluation"):
            prompt = test["prompt"]
            expected = test.get("expected", "")
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=0.1
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated[len(prompt):].strip()
            
            # Simple quality check (could be more sophisticated)
            quality_score = 1.0 if expected.lower() in generated.lower() else 0.0
            
            results.append({
                "prompt": prompt,
                "generated": generated,
                "expected": expected,
                "score": quality_score
            })
        
        avg_score = np.mean([r["score"] for r in results])
        
        return {
            "average_score": avg_score,
            "total_tests": len(test_cases),
            "passed": sum(r["score"] > 0.5 for r in results),
            "results": results[:5]  # Sample results
        }
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        self.load_model()
        
        # Test prompts
        speed_prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "How does blockchain technology work?",
            "What are the benefits of machine learning?",
            "Describe the future of technology."
        ]
        
        # Quality test cases
        quality_tests = [
            {
                "prompt": "What is Hanzo AI?",
                "expected": "frontier AI"
            },
            {
                "prompt": "What is Zoo Labs?",
                "expected": "blockchain"
            },
            {
                "prompt": "What is Lux blockchain?",
                "expected": "consensus"
            }
        ]
        
        print("\n" + "="*50)
        print(f"Benchmarking: {self.model_path}")
        print("="*50)
        
        # Run benchmarks
        speed_results = self.benchmark_inference_speed(speed_prompts)
        memory_results = self.benchmark_memory_usage()
        quality_results = self.benchmark_quality(quality_tests)
        
        # Compile results
        benchmark_results = {
            "model": self.model_path,
            "device": self.device,
            "inference_speed": speed_results,
            "memory_usage": memory_results,
            "quality": quality_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Zen models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["zen-nano"],
        help="Models to benchmark"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run benchmarks on"
    )
    
    args = parser.parse_args()
    
    all_results = {}
    
    for model_name in args.models:
        model_path = Path(args.checkpoint_dir) / model_name
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue
        
        benchmark = ZenBenchmark(str(model_path), device=args.device)
        results = benchmark.run_full_benchmark()
        all_results[model_name] = results
        
        # Print summary
        print(f"\n{model_name} Results:")
        print(f"  Inference Speed: {results['inference_speed']['tokens_per_second']:.2f} tokens/sec")
        if "peak_memory_gb" in results['memory_usage']:
            print(f"  Peak Memory: {results['memory_usage']['peak_memory_gb']:.2f} GB")
        print(f"  Quality Score: {results['quality']['average_score']:.2%}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nBenchmark results saved to {args.output}")


if __name__ == "__main__":
    main()