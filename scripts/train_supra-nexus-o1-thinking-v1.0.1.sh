#!/bin/bash
# Train supra-nexus-o1-thinking-v1.0.1 with recursive improvements

cd ~/work/zoo/gym

# Register dataset
python -c "
import json
info = json.load(open('data/dataset_info.json'))
info['supra-nexus-o1-thinking-v1.0.1_training'] = {
    'file_name': 'supra-nexus-o1-thinking-v1.0.1_training.json',
    'formatting': 'alpaca'
}
json.dump(info, open('data/dataset_info.json', 'w'), indent=2)
"

# Copy training data
cp ../../zen/data/supra-nexus-o1-thinking-v1.0.1_training.json data/

# Train with improvements
python src/train.py \
    --stage sft \
    --model_name_or_path Supra-Nexus/supra-nexus-o1-instruct \
    --dataset supra-nexus-o1-thinking-v1.0.1_training \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir output/supra-nexus-o1-thinking-v1.0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --max_steps 100 \
    --logging_steps 10 \
    --save_steps 50 \
    --do_train

# Export merged model
python src/export.py \
    --model_name_or_path Supra-Nexus/supra-nexus-o1-instruct \
    --adapter_name_or_path output/supra-nexus-o1-thinking-v1.0.1 \
    --export_dir models/supra-nexus-o1-thinking-v1.0.1

echo "✅ Training complete for supra-nexus-o1-thinking-v1.0.1"
