#!/bin/bash

echo "🚀 Pushing all Zen models to Hugging Face..."

# Function to push a model
push_model() {
    local deployment_dir=$1
    local repo_name=$2
    
    echo "📦 Pushing $repo_name..."
    
    if [ -d "$deployment_dir" ]; then
        cd "$deployment_dir"
        # Find the model subdirectory
        if [ -d "model" ]; then
            cd model
        elif [ -d "models" ]; then
            cd models/$(basename $repo_name)
        fi
        
        huggingface-cli upload $repo_name . --repo-type model 2>&1 | tail -2
        echo "✅ $repo_name pushed"
    else
        echo "⚠️  $deployment_dir not found"
    fi
    echo ""
}

cd /Users/z/work/zen

# Push zen-nano variants
push_model "zen-nano-deployment/models/zen-nano-instruct" "zenlm/zen-nano-instruct"
push_model "zen-nano-deployment/models/zen-nano-thinking" "zenlm/zen-nano-thinking"

# Push zen-omni variants  
push_model "zen-omni-deployment/models/zen-omni-thinking" "zenlm/zen-omni-thinking"
push_model "zen-omni-deployment/models/zen-omni-talking" "zenlm/zen-omni-talking"
push_model "zen-omni-deployment/models/zen-omni-captioner" "zenlm/zen-omni-captioner"

# Push other models
push_model "zen-coder-deployment/model" "zenlm/zen-coder"
push_model "zen-next-deployment" "zenlm/zen-next"
push_model "zen-3d-deployment" "zenlm/zen-3d"

echo "✨ All models pushed to Hugging Face!"
echo "Visit: https://huggingface.co/zenlm"