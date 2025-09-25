import json

with open('zen_nano_instruct_training_data.jsonl', 'r') as f:
    data = f.read()

with open('zen_nano_instruct_training_data_fixed.jsonl', 'w') as f:
    for item in data.strip().split('\n'):
        f.write(item + '\n')