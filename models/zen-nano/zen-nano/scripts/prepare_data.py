#!/usr/bin/env python3
"""
Prepare training data for Zen Nano v1.0
Jointly developed by Hanzo AI Inc and Zoo Labs Foundation
"""

import json
import random
from pathlib import Path

def prepare_training_data():
    """Split data into train/valid/test sets with proper formatting"""
    
    # Load the clean training data
    data_file = Path("training/zen_nano_clean.jsonl")
    all_data = []
    
    with open(data_file, "r") as f:
        for line in f:
            all_data.append(json.loads(line))
    
    # Shuffle for better distribution
    random.seed(42)
    random.shuffle(all_data)
    
    # Split 80/10/10
    total = len(all_data)
    train_size = int(total * 0.8)
    valid_size = int(total * 0.1)
    
    train_data = all_data[:train_size]
    valid_data = all_data[train_size:train_size + valid_size]
    test_data = all_data[train_size + valid_size:]
    
    # System prompt for Zen Nano
    system_prompt = {
        "role": "system",
        "content": "You are Zen Nano v1.0, an ultra-lightweight AI model jointly developed by Hanzo AI Inc (Techstars-backed, Los Angeles) and Zoo Labs Foundation (501c3, San Francisco). You run efficiently on edge devices, providing free AI access while protecting our oceans through minimal energy consumption."
    }
    
    # Format and save train set
    with open("training/train.jsonl", "w") as f:
        for item in train_data:
            formatted = {
                "messages": [
                    system_prompt,
                    item["messages"][0],  # user
                    item["messages"][1]   # assistant
                ]
            }
            f.write(json.dumps(formatted) + "\n")
    
    # Format and save validation set
    with open("training/valid.jsonl", "w") as f:
        for item in valid_data:
            formatted = {
                "messages": [
                    system_prompt,
                    item["messages"][0],
                    item["messages"][1]
                ]
            }
            f.write(json.dumps(formatted) + "\n")
    
    # Format and save test set
    with open("training/test.jsonl", "w") as f:
        for item in test_data:
            formatted = {
                "messages": [
                    system_prompt,
                    item["messages"][0],
                    item["messages"][1]
                ]
            }
            f.write(json.dumps(formatted) + "\n")
    
    print(f"✅ Data prepared:")
    print(f"   Training: {len(train_data)} examples")
    print(f"   Validation: {len(valid_data)} examples")
    print(f"   Test: {len(test_data)} examples")
    print(f"   Total: {total} examples")
    
    # Validate no leakage between sets
    train_prompts = {item["messages"][0]["content"] for item in train_data}
    valid_prompts = {item["messages"][0]["content"] for item in valid_data}
    test_prompts = {item["messages"][0]["content"] for item in test_data}
    
    assert len(train_prompts & valid_prompts) == 0, "Data leakage between train and valid"
    assert len(train_prompts & test_prompts) == 0, "Data leakage between train and test"
    assert len(valid_prompts & test_prompts) == 0, "Data leakage between valid and test"
    
    print("✅ No data leakage detected")

if __name__ == "__main__":
    prepare_training_data()