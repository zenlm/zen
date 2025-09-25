import json

with open('zen_nano_instruct_training_data.jsonl', 'r') as f:
    data = f.read()

with open('zen_nano_instruct_training_data_fixed_2.jsonl', 'w') as f:
    # Manually split the string by the '}{"instruction"' delimiter
    items = data.strip().split('}{"instruction"')
    for i, item in enumerate(items):
        if i > 0:
            item = '{"instruction"' + item
        if i < len(items) - 1:
            item = item + '}'
        f.write(item + '\n')
