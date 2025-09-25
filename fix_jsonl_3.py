import json

with open('zen_nano_instruct_training_data.jsonl', 'r') as f:
    data = f.read()

with open('zen_nano_instruct_training_data_fixed_3.jsonl', 'w') as f:
    # Split the string by the delimiter '}{' and add back the curly braces
    items = data.strip().replace('}{', '}\n{')
    f.write(items)