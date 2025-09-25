#!/usr/bin/env python3
'''Evaluate v1.0.1 improvements over v1.0'''

import json
from pathlib import Path

def evaluate_improvements():
    # Load training data to see what was learned
    with open('training_data_v1.0.1.jsonl') as f:
        training_data = [json.loads(line) for line in f]
    
    # Categorize improvements
    categories = {}
    for item in training_data:
        cat = item['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item['effectiveness'])
    
    # Calculate metrics
    print("📊 v1.0.1 Improvement Metrics\n")
    print("Category            | Examples | Avg Effectiveness")
    print("-" * 50)
    
    for cat, scores in sorted(categories.items()):
        avg_score = sum(scores) / len(scores)
        print(f"{cat:18} | {len(scores):8} | {avg_score:.2%}")
    
    # Overall metrics
    all_scores = []
    for scores in categories.values():
        all_scores.extend(scores)
    
    print("\n📈 Overall Statistics:")
    print(f"  Total training examples: {len(training_data)}")
    print(f"  Average effectiveness: {sum(all_scores)/len(all_scores):.2%}")
    print(f"  High-quality examples (>90%): {sum(1 for s in all_scores if s > 0.9)}")
    
    return {
        "version": "1.1.0",
        "training_examples": len(training_data),
        "categories": list(categories.keys()),
        "avg_effectiveness": sum(all_scores) / len(all_scores)
    }

if __name__ == "__main__":
    metrics = evaluate_improvements()
    
    # Save metrics
    with open('v1.0.1_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✅ Evaluation complete! Metrics saved to v1.0.1_metrics.json")
