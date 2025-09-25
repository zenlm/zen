#!/bin/bash
# Supra Nexus O1 Complete Deployment Script
# Minimal, batteries-included deployment

set -e  # Exit on error

echo "======================================================"
echo "     Supra Nexus O1 Complete Deployment"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python3 not found${NC}"
    exit 1
fi

# Check HF CLI
if command -v hf &> /dev/null; then
    HF_CMD="hf"
    echo -e "${GREEN}✓ Using new HF CLI (hf)${NC}"
elif command -v huggingface-cli &> /dev/null; then
    HF_CMD="huggingface-cli"
    echo -e "${YELLOW}⚠ Using legacy HF CLI (huggingface-cli)${NC}"
else
    echo -e "${RED}✗ HuggingFace CLI not found${NC}"
    echo "Install: pip install huggingface-hub"
    exit 1
fi

# Check authentication
echo -n "Checking HuggingFace authentication... "
if $HF_CMD auth whoami &> /dev/null || $HF_CMD whoami &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "Please authenticate first:"
    echo "  $HF_CMD auth login"
    exit 1
fi

# Step 1: Verify models
echo ""
echo "Step 1: Verifying models"
echo "------------------------"
python3 verify_supra_models.py
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Model verification failed${NC}"
    exit 1
fi

# Step 2: Optional quantization
echo ""
echo "Step 2: Quantization (Optional)"
echo "-------------------------------"
read -p "Create quantized versions? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 quantize_supra_nexus.py
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠ Quantization failed (continuing)${NC}"
    fi
else
    echo "Skipping quantization"
fi

# Step 3: Deploy to HuggingFace
echo ""
echo "Step 3: Deploy to HuggingFace"
echo "-----------------------------"
read -p "Deploy models to HuggingFace? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 deploy_supra_nexus.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Deployment failed${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}======================================================"
    echo "     Deployment Complete!"
    echo "======================================================"
    echo ""
    echo "Models available at:"
    echo "  • https://huggingface.co/supra-nexus/supra-nexus-o1-thinking"
    echo "  • https://huggingface.co/supra-nexus/supra-nexus-o1-instruct"
    echo ""
    echo "Next steps:"
    echo "  1. Verify models on HuggingFace"
    echo "  2. Test inference with example code"
    echo "  3. Share model cards and documentation"
    echo -e "======================================================${NC}"
else
    echo "Deployment cancelled"
fi