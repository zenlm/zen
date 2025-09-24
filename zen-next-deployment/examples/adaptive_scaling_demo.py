#!/usr/bin/env python3
"""
Zen-Next Adaptive Scaling Demonstration
Shows dynamic parameter activation based on task complexity
"""

import time
import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ComputeMetrics:
    """Metrics for adaptive compute"""
    task: str
    complexity: float
    active_params: float
    latency_ms: float
    memory_mb: float
    accuracy: float

class ZenNextAdaptive:
    """Zen-Next adaptive compute simulator"""

    def __init__(self):
        self.min_params = 1e9  # 1B
        self.max_params = 13e9  # 13B
        self.metrics_log = []

    def estimate_complexity(self, prompt: str, task_type: str) -> float:
        """Estimate task complexity from prompt and type"""

        # Base complexity from task type
        task_complexity = {
            "translation": 0.2,
            "qa_simple": 0.15,
            "qa_complex": 0.6,
            "summarization": 0.4,
            "reasoning": 0.7,
            "math": 0.75,
            "code_generation": 0.8,
            "creative_writing": 0.85,
            "research": 0.9
        }

        base = task_complexity.get(task_type, 0.5)

        # Modifiers from prompt characteristics
        length_factor = min(len(prompt) / 1000, 0.2)
        vocab_diversity = len(set(prompt.split())) / max(len(prompt.split()), 1)
        special_tokens = sum(1 for c in prompt if c in "()[]{}@#$%^&*")

        # Calculate final complexity
        complexity = base + length_factor + (vocab_diversity * 0.1) + (special_tokens * 0.01)
        return min(max(complexity, 0.0), 1.0)

    def activate_parameters(self, complexity: float) -> Dict:
        """Determine active parameters based on complexity"""

        # Calculate active parameters
        active_params = self.min_params + (self.max_params - self.min_params) * complexity

        # Determine configuration
        if complexity < 0.2:
            config = {
                "tier": "minimal",
                "layers": 12,
                "hidden_dim": 768,
                "attention_heads": 12,
                "params": active_params
            }
        elif complexity < 0.4:
            config = {
                "tier": "light",
                "layers": 20,
                "hidden_dim": 1024,
                "attention_heads": 16,
                "params": active_params
            }
        elif complexity < 0.6:
            config = {
                "tier": "moderate",
                "layers": 28,
                "hidden_dim": 2048,
                "attention_heads": 32,
                "params": active_params
            }
        elif complexity < 0.8:
            config = {
                "tier": "heavy",
                "layers": 36,
                "hidden_dim": 3072,
                "attention_heads": 48,
                "params": active_params
            }
        else:
            config = {
                "tier": "full",
                "layers": 48,
                "hidden_dim": 4096,
                "attention_heads": 64,
                "params": active_params
            }

        return config

    def simulate_inference(self, prompt: str, task_type: str) -> ComputeMetrics:
        """Simulate adaptive inference"""

        # Estimate complexity
        complexity = self.estimate_complexity(prompt, task_type)

        # Activate parameters
        config = self.activate_parameters(complexity)

        # Simulate latency (roughly proportional to active params)
        base_latency = 10  # ms
        param_latency = (config["params"] / 1e9) * 15  # ms per billion params
        latency = base_latency + param_latency + np.random.normal(0, 2)

        # Simulate memory usage
        memory_mb = (config["params"] / 1e6) * 2 + config["layers"] * 10

        # Simulate accuracy (higher params generally = higher accuracy, with diminishing returns)
        base_accuracy = 0.7
        param_bonus = np.log(config["params"] / 1e9) * 0.05
        noise = np.random.normal(0, 0.02)
        accuracy = min(base_accuracy + param_bonus + noise, 0.99)

        # Create metrics
        metrics = ComputeMetrics(
            task=task_type,
            complexity=complexity,
            active_params=config["params"],
            latency_ms=latency,
            memory_mb=memory_mb,
            accuracy=accuracy
        )

        self.metrics_log.append(metrics)

        return metrics, config

def demonstrate_adaptive_scaling():
    """Demonstrate adaptive scaling across different tasks"""

    print("=" * 80)
    print("ZEN-NEXT ADAPTIVE COMPUTE DEMONSTRATION")
    print("=" * 80)
    print()

    model = ZenNextAdaptive()

    # Test cases with varying complexity
    test_cases = [
        ("What is 2+2?", "qa_simple"),
        ("Translate 'Hello world' to Spanish", "translation"),
        ("Summarize this paragraph about quantum computing", "summarization"),
        ("Explain the Riemann hypothesis", "qa_complex"),
        ("Solve this differential equation: dy/dx + 2y = sin(x)", "math"),
        ("Write a recursive Fibonacci function in Python", "code_generation"),
        ("Prove that P != NP using category theory", "reasoning"),
        ("Write a sonnet about artificial intelligence", "creative_writing"),
        ("Design a distributed system for real-time ML inference at scale", "research")
    ]

    results = []

    print("TASK EXECUTION:")
    print("-" * 80)

    for prompt, task_type in test_cases:
        print(f"\nTask: {task_type}")
        print(f"Prompt: {prompt[:50]}...")

        # Run inference
        start = time.time()
        metrics, config = model.simulate_inference(prompt, task_type)

        # Display results
        print(f"  Complexity: {metrics.complexity:.3f}")
        print(f"  Active Tier: {config['tier']}")
        print(f"  Active Parameters: {metrics.active_params/1e9:.2f}B")
        print(f"  Layers: {config['layers']}, Hidden: {config['hidden_dim']}, Heads: {config['attention_heads']}")
        print(f"  Latency: {metrics.latency_ms:.1f}ms")
        print(f"  Memory: {metrics.memory_mb:.1f}MB")
        print(f"  Accuracy: {metrics.accuracy:.1%}")

        results.append(metrics)

    # Compute statistics
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY:")
    print("-" * 80)

    avg_params = np.mean([r.active_params for r in results])
    avg_latency = np.mean([r.latency_ms for r in results])
    avg_memory = np.mean([r.memory_mb for r in results])
    avg_accuracy = np.mean([r.accuracy for r in results])

    print(f"Average Active Parameters: {avg_params/1e9:.2f}B")
    print(f"Average Latency: {avg_latency:.1f}ms")
    print(f"Average Memory: {avg_memory:.1f}MB")
    print(f"Average Accuracy: {avg_accuracy:.1%}")

    # Compare to fixed model
    fixed_7b_latency = 7 * 15 + 10  # Fixed 7B model latency
    fixed_7b_memory = 7000 * 2 + 32 * 10  # Fixed 7B model memory

    print(f"\nVS FIXED 7B MODEL:")
    print(f"  Compute Savings: {(1 - avg_params/7e9)*100:.1f}%")
    print(f"  Latency Reduction: {(1 - avg_latency/fixed_7b_latency)*100:.1f}%")
    print(f"  Memory Savings: {(1 - avg_memory/fixed_7b_memory)*100:.1f}%")

    # Complexity distribution
    print("\n" + "=" * 80)
    print("COMPLEXITY DISTRIBUTION:")
    print("-" * 80)

    complexity_bins = [0.2, 0.4, 0.6, 0.8, 1.0]
    bin_names = ["Minimal", "Light", "Moderate", "Heavy", "Full"]

    for i, (threshold, name) in enumerate(zip(complexity_bins, bin_names)):
        count = sum(1 for r in results if (
            r.complexity <= threshold and
            (i == 0 or r.complexity > complexity_bins[i-1])
        ))
        percentage = (count / len(results)) * 100
        bar = "█" * int(percentage / 5)
        print(f"  {name:10s}: {bar:20s} {count}/{len(results)} ({percentage:.0f}%)")

    # Efficiency frontier
    print("\n" + "=" * 80)
    print("EFFICIENCY FRONTIER:")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.complexity):
        efficiency = r.accuracy / (r.latency_ms / 100)  # Accuracy per 100ms
        bar = "▓" * int(efficiency * 20)
        print(f"  {r.task[:15]:15s}: {bar:20s} Eff={efficiency:.2f}")

def demonstrate_memory_consolidation():
    """Demonstrate memory consolidation across sessions"""

    print("\n" + "=" * 80)
    print("MEMORY CONSOLIDATION DEMONSTRATION")
    print("=" * 80)
    print()

    class MemorySystem:
        def __init__(self):
            self.short_term = []
            self.working = []
            self.long_term = {}

        def encode(self, content: str, importance: float):
            """Encode content into short-term memory"""
            entry = {
                "content": content,
                "importance": importance,
                "timestamp": time.time(),
                "access_count": 1
            }
            self.short_term.append(entry)
            return entry

        def consolidate(self):
            """Consolidate memories from short-term to long-term"""
            consolidated = []

            for entry in self.short_term:
                if entry["importance"] > 0.5:
                    # Move to working memory
                    self.working.append(entry)

                    if entry["importance"] > 0.7:
                        # Also store in long-term
                        key = hash(entry["content"]) % 10000
                        self.long_term[key] = entry
                        consolidated.append(entry)

            # Clear short-term
            self.short_term = []
            return consolidated

        def recall(self, query: str) -> List[Dict]:
            """Recall relevant memories"""
            results = []

            # Search all memory stores
            for memory in self.short_term + self.working + list(self.long_term.values()):
                if query.lower() in memory["content"].lower():
                    memory["access_count"] += 1
                    results.append(memory)

            return results

    memory = MemorySystem()

    # Simulate sessions
    sessions = [
        ("Session 1", [
            ("User's name is Alice", 0.9),
            ("Prefers Python over Java", 0.7),
            ("Working on ML project", 0.8),
            ("Likes coffee", 0.3)
        ]),
        ("Session 2", [
            ("Alice's project uses TensorFlow", 0.8),
            ("Deadline is next Friday", 0.6),
            ("Team size is 5 people", 0.5),
            ("Office is in building A", 0.2)
        ]),
        ("Session 3", [
            ("Alice completed the ML model", 0.9),
            ("Accuracy reached 94%", 0.8),
            ("Presented to stakeholders", 0.7),
            ("Weather was sunny", 0.1)
        ])
    ]

    for session_name, memories in sessions:
        print(f"\n{session_name}:")
        print("-" * 40)

        for content, importance in memories:
            entry = memory.encode(content, importance)
            print(f"  Encoded: {content[:30]:30s} (importance={importance:.1f})")

        # Consolidate after each session
        consolidated = memory.consolidate()
        print(f"  Consolidated: {len(consolidated)} memories to long-term")

    # Test recall
    print("\n" + "=" * 80)
    print("MEMORY RECALL TESTS:")
    print("-" * 80)

    queries = ["Alice", "ML", "project", "TensorFlow", "coffee", "weather"]

    for query in queries:
        results = memory.recall(query)
        print(f"\nQuery: '{query}'")
        print(f"  Found: {len(results)} memories")
        for r in results[:3]:  # Show top 3
            print(f"    - {r['content'][:50]:50s} (accessed {r['access_count']}x)")

    # Show memory statistics
    print("\n" + "=" * 80)
    print("MEMORY STATISTICS:")
    print("-" * 80)
    print(f"  Short-term: {len(memory.short_term)} entries")
    print(f"  Working: {len(memory.working)} entries")
    print(f"  Long-term: {len(memory.long_term)} entries")

    # Calculate retention
    total_encoded = sum(len(mems) for _, mems in sessions)
    total_retained = len(memory.working) + len(memory.long_term)
    retention_rate = (total_retained / total_encoded) * 100

    print(f"  Retention rate: {retention_rate:.1f}%")

def demonstrate_architecture_evolution():
    """Demonstrate neural architecture search evolution"""

    print("\n" + "=" * 80)
    print("NEURAL ARCHITECTURE EVOLUTION DEMONSTRATION")
    print("=" * 80)
    print()

    class Architecture:
        def __init__(self):
            self.genome = {
                "attention_pattern": "global",
                "ffn_type": "standard",
                "layer_norm": "pre",
                "activation": "gelu",
                "dropout": 0.1
            }
            self.fitness = 0.5

        def mutate(self):
            """Apply random mutation"""
            import random

            mutations = {
                "attention_pattern": ["global", "local", "sparse", "dilated"],
                "ffn_type": ["standard", "gated", "mixture"],
                "layer_norm": ["pre", "post", "both"],
                "activation": ["gelu", "swish", "relu", "silu"],
                "dropout": lambda: random.uniform(0.0, 0.3)
            }

            # Random mutation
            gene = random.choice(list(self.genome.keys()))
            if gene == "dropout":
                self.genome[gene] = mutations[gene]()
            else:
                self.genome[gene] = random.choice(mutations[gene])

        def evaluate(self) -> float:
            """Evaluate fitness"""
            # Simulated fitness based on architecture
            scores = {
                "attention_pattern": {"global": 0.8, "local": 0.7, "sparse": 0.9, "dilated": 0.75},
                "ffn_type": {"standard": 0.7, "gated": 0.85, "mixture": 0.9},
                "layer_norm": {"pre": 0.8, "post": 0.7, "both": 0.85},
                "activation": {"gelu": 0.8, "swish": 0.85, "relu": 0.7, "silu": 0.82}
            }

            fitness = 1.0
            for gene, value in self.genome.items():
                if gene == "dropout":
                    # Lower dropout generally better
                    fitness *= (1 - value * 0.5)
                else:
                    fitness *= scores[gene].get(value, 0.5)

            self.fitness = fitness
            return fitness

    # Initialize population
    population_size = 10
    population = [Architecture() for _ in range(population_size)]

    # Evolution
    generations = 10
    history = []

    print("EVOLUTION PROGRESS:")
    print("-" * 80)

    for gen in range(generations):
        # Evaluate fitness
        for arch in population:
            arch.evaluate()

        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Record best
        best = population[0]
        avg_fitness = np.mean([a.fitness for a in population])
        history.append((best.fitness, avg_fitness))

        print(f"\nGeneration {gen + 1}:")
        print(f"  Best fitness: {best.fitness:.4f}")
        print(f"  Average fitness: {avg_fitness:.4f}")
        print(f"  Best genome: {json.dumps(best.genome, indent=4)}")

        # Selection and reproduction
        survivors = population[:population_size // 2]
        offspring = []

        for parent in survivors:
            child = Architecture()
            child.genome = parent.genome.copy()
            child.mutate()
            offspring.append(child)

        population = survivors + offspring

    # Show evolution trajectory
    print("\n" + "=" * 80)
    print("EVOLUTION TRAJECTORY:")
    print("-" * 80)

    for i, (best, avg) in enumerate(history):
        best_bar = "█" * int(best * 30)
        avg_bar = "░" * int(avg * 30)
        print(f"  Gen {i+1:2d}: {best_bar:30s} Best={best:.3f} Avg={avg:.3f}")

    # Final architecture
    final_best = population[0]
    print("\n" + "=" * 80)
    print("FINAL EVOLVED ARCHITECTURE:")
    print("-" * 80)
    print(json.dumps(final_best.genome, indent=2))
    print(f"Fitness: {final_best.fitness:.4f}")
    print(f"Improvement: {(final_best.fitness - history[0][0]) / history[0][0] * 100:.1f}%")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_adaptive_scaling()
    demonstrate_memory_consolidation()
    demonstrate_architecture_evolution()

    print("\n" + "=" * 80)
    print("ZEN-NEXT EXPERIMENTAL FEATURES DEMONSTRATED")
    print("These capabilities will be refined for Zen2 production release")
    print("=" * 80)