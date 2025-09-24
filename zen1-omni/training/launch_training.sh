#!/bin/bash

# Zen1-Omni HuggingFace Training Launch Script
# Distributed training with DeepSpeed and GSPO

set -e

# Default Configuration (can be overridden by environment variables)
: "${MODEL_NAME:=Qwen/Qwen3-Omni-30B-A3B-Instruct}"
: "${DATASET_NAME:=zen-ai/zen1-omni-preferences}"
: "${OUTPUT_DIR:=zen-ai/zen1-omni-30b-gspo}"
: "${NUM_GPUS:=8}"
: "${BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION:=16}"
: "${LEARNING_RATE:=5e-5}"
: "${NUM_EPOCHS:=3}"
: "${LORA_RANK:=128}"

# Check for HuggingFace Token
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please set it to your HuggingFace API token."
    exit 1
fi

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "Zen1-Omni Training on HuggingFace"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "=================================================="

# Function to upload dataset
upload_dataset() {
    echo "Uploading dataset to HuggingFace Hub..."
    python $SCRIPT_DIR/dataset_upload.py \
        --data_dir $PROJECT_DIR/data/preferences \
        --dataset_type preference \
        --repo_id $DATASET_NAME \
        --token $HF_TOKEN
}

# Function to train with GSPO
train_gspo() {
    echo "Starting GSPO training..."

    if [ $NUM_GPUS -gt 1 ]; then
        # Multi-GPU training with DeepSpeed
        deepspeed --num_gpus=$NUM_GPUS \
            $SCRIPT_DIR/train_zen1_gspo.py \
            --model_name $MODEL_NAME \
            --dataset_name $DATASET_NAME \
            --hub_model_id $OUTPUT_DIR \
            --num_epochs $NUM_EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --lora_r $LORA_RANK \
            --deepspeed $SCRIPT_DIR/deepspeed_zen1.json \
            --push_to_hub \
            --use_wandb
    else
        # Single GPU training
        python $SCRIPT_DIR/train_zen1_gspo.py \
            --model_name $MODEL_NAME \
            --dataset_name $DATASET_NAME \
            --hub_model_id $OUTPUT_DIR \
            --num_epochs $NUM_EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --lora_r $LORA_RANK \
            --push_to_hub \
            --use_wandb
    fi
}

# Function to train with AutoTrain
train_autotrain() {
    echo "Starting AutoTrain..."

    autotrain llm \
        --model $MODEL_NAME \
        --project-name zen1-omni-gspo \
        --data-path $DATASET_NAME \
        --text-column text \
        --trainer dpo \
        --batch-size $BATCH_SIZE \
        --epochs $NUM_EPOCHS \
        --lr $LEARNING_RATE \
        --peft \
        --quantization int4 \
        --lora-r $LORA_RANK \
        --lora-alpha $((LORA_RANK * 2)) \
        --target-modules all-linear \
        --push-to-hub \
        --hub-model-id $OUTPUT_DIR \
        --hub-token $HF_TOKEN
}

# Function to deploy to Spaces
deploy_spaces() {
    echo "Deploying to HuggingFace Spaces..."

    # Create Spaces repository
    huggingface-cli repo create zen1-omni-demo \
        --type space \
        --space-sdk gradio \
        --private false

    # Clone and setup
    git clone https://huggingface.co/spaces/$BRAND_NAME/zen1-omni-demo
    cp $SCRIPT_DIR/spaces_app.py zen1-omni-demo/app.py
    cp $SCRIPT_DIR/requirements_spaces.txt zen1-omni-demo/requirements.txt

    # Add README
    cat > zen1-omni-demo/README.md << EOF
---
title: Zen1-Omni Multimodal Demo
emoji: 🌟
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: true
models:
  - $OUTPUT_DIR
---

# Zen1-Omni Multimodal AI Demo

Interactive demo for Zen1-Omni, supporting text, audio, image, and video understanding.
EOF

    # Push to Spaces
    cd zen1-omni-demo
    git add .
    git commit -m "Deploy Zen1-Omni demo"
    git push
    cd ..

    echo "✓ Deployed to: https://huggingface.co/spaces/$BRAND_NAME/zen1-omni-demo"
}

# Function to run evaluation
evaluate() {
    echo "Running evaluation..."

    python -m lm_eval \
        --model hf \
        --model_args pretrained=$OUTPUT_DIR,peft=$OUTPUT_DIR \
        --tasks mmlu,gsm8k,hellaswag \
        --device cuda \
        --batch_size 1 \
        --output_path $PROJECT_DIR/eval_results
}

# Main execution
main() {
    case "${1:-gspo}" in
        upload)
            upload_dataset
            ;; 
        gspo)
            train_gspo
            ;; 
        autotrain)
            train_autotrain
            ;; 
        deploy)
            deploy_spaces
            ;; 
        evaluate)
            evaluate
            ;; 
        full)
            upload_dataset
            train_gspo
            evaluate
            deploy_spaces
            ;; 
        *)
            echo "Usage: $0 {upload|gspo|autotrain|deploy|evaluate|full}"
            exit 1
            ;; 
    esac
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    echo "No arguments provided, running GSPO training..."
    main gspo
else
    main "$@"
fi

echo "=================================================="
echo "Zen1-Omni training completed successfully!"
echo "Model available at: https://huggingface.co/$OUTPUT_DIR"
echo "=================================================="
