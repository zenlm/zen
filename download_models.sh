#!/bin/bash

# Download Qwen3 base models for Zen family
# Uses git-lfs and huggingface-cli

set -e

BASE_DIR="/Users/z/work/zen/base-models"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Model configurations (using arrays for compatibility)
MODELS_KEYS=(
    "zen-nano-instruct"
    "zen-nano-thinking"
    "zen-omni"
    "zen-omni-thinking"
    "zen-omni-captioner"
)

MODELS_VALUES=(
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-4B-Thinking-2507"
    "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    "Qwen/Qwen3-Omni-30B-A3B-Captioner"
)

# Helper function to get model repo
get_model_repo() {
    local variant=$1
    for i in "${!MODELS_KEYS[@]}"; do
        if [[ "${MODELS_KEYS[$i]}" == "$variant" ]]; then
            echo "${MODELS_VALUES[$i]}"
            return 0
        fi
    done
    return 1
}

# Function to check dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    # Check for git-lfs
    if ! command -v git-lfs &> /dev/null; then
        echo -e "${YELLOW}git-lfs not found. Installing...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install git-lfs
        else
            echo -e "${RED}Please install git-lfs manually${NC}"
            exit 1
        fi
        git lfs install
    fi
    
    # Check for Python and pip
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python3 is required${NC}"
        exit 1
    fi
    
    # Install huggingface-hub if not present
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        echo -e "${YELLOW}Installing huggingface-hub...${NC}"
        pip3 install --user huggingface_hub
    fi
    
    echo -e "${GREEN}Dependencies OK${NC}"
}

# Function to download model using git-lfs
download_with_git() {
    local variant=$1
    local repo_id=$2
    local local_dir="$BASE_DIR/${repo_id#*/}"
    
    echo -e "\n${BLUE}Downloading $variant via git-lfs${NC}"
    echo -e "  Repository: $repo_id"
    echo -e "  Local path: $local_dir"
    
    if [[ -d "$local_dir/.git" ]]; then
        echo -e "${YELLOW}Repository exists, pulling latest...${NC}"
        cd "$local_dir"
        git pull
        git lfs pull
    else
        mkdir -p "$BASE_DIR"
        cd "$BASE_DIR"
        git clone "https://huggingface.co/$repo_id" "${repo_id#*/}"
        cd "${repo_id#*/}"
        git lfs pull
    fi
    
    echo -e "${GREEN}✓ Downloaded $variant${NC}"
}

# Function to download using huggingface-cli
download_with_hf_cli() {
    local variant=$1
    local repo_id=$2
    local local_dir="$BASE_DIR/${repo_id#*/}"
    
    echo -e "\n${BLUE}Downloading $variant via huggingface-cli${NC}"
    echo -e "  Repository: $repo_id"
    echo -e "  Local path: $local_dir"
    
    mkdir -p "$local_dir"
    
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$repo_id',
    local_dir='$local_dir',
    local_dir_use_symlinks=False,
    resume_download=True
)
"
    
    echo -e "${GREEN}✓ Downloaded $variant${NC}"
}

# Function to verify download
verify_download() {
    local variant=$1
    local repo_id=$2
    local local_dir="$BASE_DIR/${repo_id#*/}"
    
    echo -e "\n${BLUE}Verifying $variant...${NC}"
    
    if [[ ! -d "$local_dir" ]]; then
        echo -e "${RED}✗ Directory not found: $local_dir${NC}"
        return 1
    fi
    
    # Check for essential files
    local missing=""
    for file in config.json tokenizer.json tokenizer_config.json; do
        if [[ ! -f "$local_dir/$file" ]]; then
            missing="$missing $file"
        fi
    done
    
    if [[ -n "$missing" ]]; then
        echo -e "${YELLOW}⚠ Missing files:$missing${NC}"
    fi
    
    # Check for model files
    local model_count=$(find "$local_dir" -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \) | wc -l)
    
    if [[ $model_count -eq 0 ]]; then
        echo -e "${RED}✗ No model files found${NC}"
        return 1
    fi
    
    # Calculate total size
    local total_size=$(du -sh "$local_dir" | cut -f1)
    
    echo -e "${GREEN}✓ Verified: $model_count model files, Total size: $total_size${NC}"
    return 0
}

# Function to list models
list_models() {
    echo -e "\n${BLUE}Available models for Zen family:${NC}"
    echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for i in "${!MODELS_KEYS[@]}"; do
        local variant="${MODELS_KEYS[$i]}"
        local repo="${MODELS_VALUES[$i]}"
        echo -e "${GREEN}$variant${NC}"
        echo -e "  Repo: $repo"
        local local_dir="$BASE_DIR/${repo#*/}"
        if [[ -d "$local_dir" ]]; then
            local size=$(du -sh "$local_dir" 2>/dev/null | cut -f1)
            echo -e "  Status: ${GREEN}Downloaded (${size})${NC}"
        else
            echo -e "  Status: ${YELLOW}Not downloaded${NC}"
        fi
        echo
    done
}

# Main function
main() {
    local action="download"
    local method="hf"  # Default to huggingface-hub
    local models_to_download=()
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --list)
                action="list"
                shift
                ;;
            --verify)
                action="verify"
                shift
                ;;
            --git)
                method="git"
                shift
                ;;
            --hf)
                method="hf"
                shift
                ;;
            --all)
                models_to_download=("${MODELS_KEYS[@]}")
                shift
                ;;
            *)
                if get_model_repo "$1" >/dev/null; then
                    models_to_download+=("$1")
                else
                    echo -e "${RED}Unknown model: $1${NC}"
                    echo "Available models: ${MODELS_KEYS[@]}"
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Default to all models if none specified
    if [[ ${#models_to_download[@]} -eq 0 ]] && [[ "$action" == "download" ]]; then
        models_to_download=("${MODELS_KEYS[@]}")
    fi
    
    # Execute action
    case $action in
        list)
            list_models
            ;;
        verify)
            for variant in "${models_to_download[@]}"; do
                repo=$(get_model_repo "$variant")
                verify_download "$variant" "$repo"
            done
            ;;
        download)
            check_dependencies
            
            echo -e "\n${BLUE}Downloading ${#models_to_download[@]} models...${NC}"
            echo -e "Method: $method"
            echo -e "Base directory: $BASE_DIR"
            
            for variant in "${models_to_download[@]}"; do
                repo=$(get_model_repo "$variant")
                if [[ "$method" == "git" ]]; then
                    download_with_git "$variant" "$repo"
                else
                    download_with_hf_cli "$variant" "$repo"
                fi
                verify_download "$variant" "$repo"
            done
            
            echo -e "\n${GREEN}Download complete!${NC}"
            ;;
    esac
}

# Show usage if no arguments
if [[ $# -eq 0 ]]; then
    cat << EOF
Usage: $0 [OPTIONS] [MODELS...]

Download Qwen3 base models for Zen family

OPTIONS:
    --list      List available models and their status
    --verify    Verify downloaded models
    --git       Use git-lfs for download (default: huggingface-hub)
    --hf        Use huggingface-hub for download (default)
    --all       Download all models

MODELS:
    zen-nano-instruct     Qwen3-4B-Instruct-2507
    zen-nano-thinking     Qwen3-4B-Thinking-2507
    zen-omni             Qwen3-Omni-30B-A3B-Instruct
    zen-omni-thinking    Qwen3-Omni-30B-A3B-Thinking
    zen-omni-captioner   Qwen3-Omni-30B-A3B-Captioner

EXAMPLES:
    $0 --all                    # Download all models
    $0 zen-nano-instruct        # Download specific model
    $0 --list                   # List models and status
    $0 --verify zen-omni        # Verify specific model
EOF
    exit 0
fi

main "$@"