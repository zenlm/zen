#!/usr/bin/env python3
"""Simple Zen Nano finetuning script"""

import json
from pathlib import Path

# Prepare training data
training_data = [
    {"instruction": "What is Zen Nano?", "response": "Zen Nano is an ultra-lightweight AI model optimized for edge deployment."},
    {"instruction": "Who created Zen?", "response": "Zen was created by Hanzo AI, focusing on efficient and powerful AI systems."},
    {"instruction": "What makes Zen special?", "response": "Zen combines efficiency with capability, offering models from nano to omni scale."},
]

# Save training data
data_path = Path("zen-nano/data/finetune_data.jsonl")
data_path.parent.mkdir(exist_ok=True)

with open(data_path, "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")

print(f"✅ Created training data at {data_path}")
print(f"   {len(training_data)} examples ready for finetuning")
