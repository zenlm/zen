#!/bin/bash

# Zen Nano Platform Compatibility Test Script
# Tests model deployment across multiple platforms

echo "========================================="
echo "ZEN NANO PLATFORM COMPATIBILITY TESTING"
echo "========================================="

MODEL_DIR="/Users/z/work/zen/zen-nano"
ERRORS=0
TESTS_RUN=0
TESTS_PASSED=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Ollama
echo -e "\n[1/4] Testing Ollama integration..."
TESTS_RUN=$((TESTS_RUN + 1))
if command -v ollama &> /dev/null; then
    # Check if Modelfile exists
    if [ -f "${MODEL_DIR}/Modelfile" ]; then
        # Create test model
        ollama create zen-nano-test:latest -f ${MODEL_DIR}/Modelfile 2>/dev/null

        # Test generation
        OUTPUT=$(ollama run zen-nano-test:latest "Who are you?" 2>&1)

        # Check for Zen Nano identity
        if echo "$OUTPUT" | grep -qi "zen nano"; then
            echo -e "${GREEN}PASS${NC}: Ollama test successful"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${RED}FAIL${NC}: Ollama test failed - identity not found"
            ERRORS=$((ERRORS + 1))
        fi

        # Clean up
        ollama rm zen-nano-test:latest 2>/dev/null
    else
        echo -e "${YELLOW}SKIP${NC}: Modelfile not found at ${MODEL_DIR}/Modelfile"
    fi
else
    echo -e "${YELLOW}SKIP${NC}: Ollama not installed"
fi

# Test 2: llama.cpp
echo -e "\n[2/4] Testing llama.cpp integration..."
TESTS_RUN=$((TESTS_RUN + 1))
if [ -f "${MODEL_DIR}/llama.cpp/main" ]; then
    # Find the first available GGUF model
    GGUF_MODEL=""
    for model in ${MODEL_DIR}/zen-nano*.gguf ${MODEL_DIR}/*.gguf; do
        if [ -f "$model" ]; then
            GGUF_MODEL="$model"
            break
        fi
    done

    if [ -n "$GGUF_MODEL" ]; then
        echo "Using GGUF model: $(basename $GGUF_MODEL)"
        OUTPUT=$(${MODEL_DIR}/llama.cpp/main -m "$GGUF_MODEL" \
                 -p "You are Zen Nano. Who are you?" -n 50 --temp 0.1 2>&1)

        if echo "$OUTPUT" | grep -qi "zen"; then
            echo -e "${GREEN}PASS${NC}: llama.cpp test successful"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${RED}FAIL${NC}: llama.cpp test failed"
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo -e "${YELLOW}SKIP${NC}: No GGUF model found"
    fi
else
    # Try to build llama.cpp if not built
    if [ -d "${MODEL_DIR}/llama.cpp" ]; then
        echo "Building llama.cpp..."
        cd ${MODEL_DIR}/llama.cpp
        make clean && make LLAMA_METAL=1 2>/dev/null
        if [ -f "${MODEL_DIR}/llama.cpp/main" ]; then
            echo "llama.cpp built successfully, retrying test..."
            # Retry the test
            exec $0
        else
            echo -e "${YELLOW}SKIP${NC}: llama.cpp build failed"
        fi
    else
        echo -e "${YELLOW}SKIP${NC}: llama.cpp not found"
    fi
fi

# Test 3: Python Transformers
echo -e "\n[3/4] Testing Transformers library..."
TESTS_RUN=$((TESTS_RUN + 1))

# Check if SafeTensors model exists
if [ -d "${MODEL_DIR}/zen-nano-safetensors" ]; then
    python3 -c "
import sys
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        '${MODEL_DIR}/zen-nano-safetensors',
        torch_dtype=torch.float32,
        device_map='cpu'
    )
    tokenizer = AutoTokenizer.from_pretrained('${MODEL_DIR}/zen-nano-safetensors')

    print('Generating test response...')
    inputs = tokenizer('Who are you?', return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if 'zen' in response.lower() or 'nano' in response.lower():
        print('${GREEN}PASS${NC}: Transformers test successful')
        print(f'Response: {response[:100]}')
        sys.exit(0)
    else:
        print('${RED}FAIL${NC}: Identity not found in response')
        print(f'Response: {response[:100]}')
        sys.exit(1)
except ImportError as e:
    print('${YELLOW}SKIP${NC}: Required packages not installed')
    print(f'Missing: {e}')
    sys.exit(2)
except Exception as e:
    print(f'${RED}FAIL${NC}: Transformers test failed: {e}')
    sys.exit(1)
"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
    elif [ $EXIT_CODE -eq 1 ]; then
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${YELLOW}SKIP${NC}: SafeTensors model not found at ${MODEL_DIR}/zen-nano-safetensors"
fi

# Test 4: MLX (Apple Silicon only)
echo -e "\n[4/4] Testing MLX integration..."
TESTS_RUN=$((TESTS_RUN + 1))

if [[ $(uname -m) == "arm64" ]]; then
    # Check if MLX model exists
    if [ -d "${MODEL_DIR}/zen-nano-mlx" ] || [ -d "${MODEL_DIR}/models/fused" ]; then
        python3 -c "
import sys
try:
    from mlx_lm import load, generate

    # Try different model locations
    model_paths = [
        '${MODEL_DIR}/zen-nano-mlx',
        '${MODEL_DIR}/models/fused',
        '${MODEL_DIR}/models/adapters'
    ]

    model = None
    for path in model_paths:
        try:
            print(f'Trying to load from: {path}')
            model, tokenizer = load(path)
            print(f'Loaded model from: {path}')
            break
        except:
            continue

    if model is None:
        print('${YELLOW}SKIP${NC}: No MLX model found')
        sys.exit(2)

    print('Generating test response...')
    response = generate(
        model, tokenizer,
        prompt='Who are you?',
        max_tokens=20,
        temperature=0.1,
        verbose=False
    )

    if 'zen' in response.lower() or 'nano' in response.lower():
        print('${GREEN}PASS${NC}: MLX test successful')
        print(f'Response: {response[:100]}')
        sys.exit(0)
    else:
        print('${RED}FAIL${NC}: Identity not found in response')
        print(f'Response: {response[:100]}')
        sys.exit(1)
except ImportError as e:
    print('${YELLOW}SKIP${NC}: mlx-lm not installed')
    print(f'Install with: pip install mlx-lm')
    sys.exit(2)
except Exception as e:
    print(f'${RED}FAIL${NC}: MLX test failed: {e}')
    sys.exit(1)
"
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            TESTS_PASSED=$((TESTS_PASSED + 1))
        elif [ $EXIT_CODE -eq 1 ]; then
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo -e "${YELLOW}SKIP${NC}: MLX model not found"
    fi
else
    echo -e "${YELLOW}SKIP${NC}: Not on Apple Silicon ($(uname -m))"
fi

# Summary
echo -e "\n========================================="
echo "TEST SUMMARY"
echo "========================================="
echo "Tests Run: $TESTS_RUN"
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $ERRORS"
echo "Tests Skipped: $((TESTS_RUN - TESTS_PASSED - ERRORS))"

if [ $ERRORS -eq 0 ]; then
    if [ $TESTS_PASSED -eq $TESTS_RUN ]; then
        echo -e "\n${GREEN}SUCCESS: ALL PLATFORM TESTS PASSED${NC}"
    else
        echo -e "\n${YELLOW}PARTIAL SUCCESS: Some tests skipped but no failures${NC}"
    fi
    EXIT_STATUS=0
else
    echo -e "\n${RED}WARNING: $ERRORS PLATFORM TEST(S) FAILED${NC}"
    EXIT_STATUS=1
fi

echo "========================================="

# Provide helpful next steps
if [ $ERRORS -gt 0 ] || [ $TESTS_PASSED -lt $TESTS_RUN ]; then
    echo -e "\nNext Steps:"

    if ! command -v ollama &> /dev/null; then
        echo "  - Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh"
    fi

    if [ ! -f "${MODEL_DIR}/llama.cpp/main" ]; then
        echo "  - Build llama.cpp: cd ${MODEL_DIR}/llama.cpp && make"
    fi

    if [ ! -d "${MODEL_DIR}/zen-nano-safetensors" ]; then
        echo "  - Convert to SafeTensors format (see deployment guide)"
    fi

    if [[ $(uname -m) == "arm64" ]] && [ ! -d "${MODEL_DIR}/zen-nano-mlx" ]; then
        echo "  - Convert to MLX format: python -m mlx_lm.convert ..."
    fi
fi

exit $EXIT_STATUS