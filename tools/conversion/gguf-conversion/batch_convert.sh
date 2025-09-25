#!/bin/bash

# GGUF Batch Conversion Script for Zen Models
# Converts all Zen models to optimized GGUF formats

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_CPP_DIR="/Users/z/work/zen/llama.cpp"
OUTPUT_DIR="$SCRIPT_DIR/output"
MODELS_DIR="/Users/z/work/zen/models"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi

    # Check llama.cpp
    if [ ! -d "$LLAMA_CPP_DIR" ]; then
        print_error "llama.cpp not found at $LLAMA_CPP_DIR"
        exit 1
    fi

    # Check conversion script
    if [ ! -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]; then
        print_error "convert_hf_to_gguf.py not found in llama.cpp"
        exit 1
    fi

    # Build llama.cpp if needed
    if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
        print_warning "llama-quantize not found, building llama.cpp..."
        build_llama_cpp
    fi

    print_success "All dependencies checked"
}

# Build llama.cpp
build_llama_cpp() {
    print_status "Building llama.cpp..."

    mkdir -p "$LLAMA_CPP_DIR/build"
    cd "$LLAMA_CPP_DIR/build"

    # Configure with Metal support for macOS
    cmake .. \
        -DLLAMA_METAL=ON \
        -DLLAMA_ACCELERATE=ON \
        -DCMAKE_BUILD_TYPE=Release

    # Build with parallel jobs
    make -j$(sysctl -n hw.ncpu)

    if [ $? -eq 0 ]; then
        print_success "llama.cpp built successfully"
    else
        print_error "Failed to build llama.cpp"
        exit 1
    fi

    cd "$SCRIPT_DIR"
}

# Convert a single model
convert_model() {
    local model_name=$1
    local model_path=$2
    local quantizations=$3

    print_status "Converting $model_name..."

    # Run Python conversion script for this model
    python3 "$SCRIPT_DIR/convert_zen_to_gguf.py" \
        --model "$model_name" \
        --llama-cpp-path "$LLAMA_CPP_DIR"

    if [ $? -eq 0 ]; then
        print_success "Successfully converted $model_name"
    else
        print_error "Failed to convert $model_name"
        return 1
    fi
}

# Main conversion pipeline
main() {
    print_status "Starting Zen Models GGUF Conversion Pipeline"
    echo "================================================"

    # Check dependencies
    check_dependencies

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Models to convert
    declare -a models=(
        "zen-nano-instruct"
        "zen-nano-thinking"
        "zen-omni"
        "zen-omni-thinking"
        "zen-omni-captioner"
        "zen-coder"
        "zen-next"
    )

    # Convert all models
    if [ "$1" == "--all" ] || [ -z "$1" ]; then
        print_status "Converting all Zen models..."
        python3 "$SCRIPT_DIR/convert_zen_to_gguf.py" --model all

    elif [ "$1" == "--parallel" ]; then
        print_status "Converting all models in parallel..."
        python3 "$SCRIPT_DIR/convert_zen_to_gguf.py" --model all

    else
        # Convert specific model
        if [[ " ${models[@]} " =~ " $1 " ]]; then
            convert_model "$1"
        else
            print_error "Unknown model: $1"
            echo "Available models: ${models[*]}"
            exit 1
        fi
    fi

    # Show results
    if [ -d "$OUTPUT_DIR" ]; then
        print_status "Conversion results:"
        echo "==================="
        ls -lah "$OUTPUT_DIR"/*.gguf 2>/dev/null || print_warning "No GGUF files generated"

        # Show summary if available
        if [ -f "$OUTPUT_DIR/conversion_summary.json" ]; then
            echo ""
            print_status "Conversion Summary:"
            python3 -m json.tool "$OUTPUT_DIR/conversion_summary.json"
        fi
    fi

    print_success "GGUF conversion pipeline complete!"
}

# Handle script arguments
case "$1" in
    --help|-h)
        echo "Usage: $0 [OPTIONS] [MODEL_NAME]"
        echo ""
        echo "Options:"
        echo "  --all        Convert all models (default)"
        echo "  --parallel   Convert models in parallel"
        echo "  --help       Show this help message"
        echo ""
        echo "Models:"
        echo "  zen-nano-instruct"
        echo "  zen-nano-thinking"
        echo "  zen-omni"
        echo "  zen-omni-thinking"
        echo "  zen-omni-captioner"
        echo "  zen-coder"
        echo "  zen-next"
        echo ""
        echo "Example:"
        echo "  $0                    # Convert all models"
        echo "  $0 zen-nano-instruct  # Convert specific model"
        echo "  $0 --parallel         # Convert all in parallel"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac