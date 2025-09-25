#!/usr/bin/env python3

import json

def combine_and_format_data():
    """Combines and formats the training data."""
    formatted_data = []

    # Process hanzo_training.jsonl
    with open("hanzo_training.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            formatted_data.append({"text": f"Human: {item['prompt']}\n\nAssistant: {item['completion']}"})

    # Process zen1_branding_data.json
    with open("zen1_branding_data.json", "r") as f:
        data = json.load(f)
        for item in data:
            formatted_data.append(item)

    # Shuffle the data
    import random
    random.shuffle(formatted_data)

    # Split the data
    split_index = int(len(formatted_data) * 0.9)
    train_data = formatted_data[:split_index]
    valid_data = formatted_data[split_index:]

    # Write to train.jsonl
    with open("zen-nano/data/train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    # Write to valid.jsonl
    with open("zen-nano/data/valid.jsonl", "w") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    combine_and_format_data()