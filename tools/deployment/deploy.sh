#!/bin/bash

# Deploy Zen models to Hugging Face
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Zen Model Deployment Script${NC}"
echo -e "${BLUE}================================${NC}\n"

# Check for HF token
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}⚠️  HF_TOKEN not set${NC}"
    echo "Please export your Hugging Face token:"
    echo "  export HF_TOKEN='your_token_here'"
    echo "Get token from: https://huggingface.co/settings/tokens"
    exit 1
fi

# Function to check if model directory exists
check_model() {
    local model_path=$1
    if [ -d "$model_path" ]; then
        echo -e "${GREEN}✅ Found: $model_path${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Missing: $model_path${NC}"
        return 1
    fi
}

# Check model directories
echo -e "${BLUE}Checking model directories...${NC}"
check_model "models/zen-nano"
check_model "models/zen-nano-instruct"
check_model "models/zen-nano-thinking"
check_model "models/zen-omni"
check_model "models/zen-coder"
check_model "models/zen-next"

echo ""

# Menu
echo -e "${BLUE}What would you like to deploy?${NC}"
echo "1) All models"
echo "2) Zen-Nano family (nano, instruct, thinking)"
echo "3) Zen-Omni (multimodal)"
echo "4) Zen-Coder"
echo "5) Zen-Next"
echo "6) Quantized versions (GGUF/MLX)"
echo "7) Custom selection"
echo "8) Dry run (test without uploading)"

read -p "Select option [1-8]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}Deploying all models...${NC}"
        python3 deploy_to_hf.py --models all
        ;;
    2)
        echo -e "\n${GREEN}Deploying Zen-Nano family...${NC}"
        python3 deploy_to_hf.py --models zen-nano zen-nano-instruct zen-nano-thinking
        ;;
    3)
        echo -e "\n${GREEN}Deploying Zen-Omni...${NC}"
        python3 deploy_to_hf.py --models zen-omni
        ;;
    4)
        echo -e "\n${GREEN}Deploying Zen-Coder...${NC}"
        python3 deploy_to_hf.py --models zen-coder
        ;;
    5)
        echo -e "\n${GREEN}Deploying Zen-Next...${NC}"
        python3 deploy_to_hf.py --models zen-next
        ;;
    6)
        echo -e "\n${GREEN}Deploying quantized versions...${NC}"
        python3 deploy_to_hf.py --models zen-nano-gguf zen-nano-mlx
        ;;
    7)
        echo -e "\n${BLUE}Available models:${NC}"
        echo "  zen-nano"
        echo "  zen-nano-instruct"
        echo "  zen-nano-thinking"
        echo "  zen-omni"
        echo "  zen-coder"
        echo "  zen-next"
        echo "  zen-nano-gguf"
        echo "  zen-nano-mlx"
        read -p "Enter model names (space-separated): " models
        python3 deploy_to_hf.py --models $models
        ;;
    8)
        echo -e "\n${YELLOW}Running dry run (no upload)...${NC}"
        python3 deploy_to_hf.py --models all --dry-run
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo -e "\n${BLUE}Deployment complete!${NC}"
echo -e "Visit: https://huggingface.co/zenlm to see your models\n"