import json

with open('zen_nano_instruct_training_data.jsonl', 'r') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error on line {i+1}: {e}")
            print(line)