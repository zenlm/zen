# Zen Model Ecosystem - Complete Reproduction Makefile
# Integrates with ~/work/zoo/gym for training
# Handles MLX + GGUF format generation and HuggingFace deployment

# Configuration
PYTHON := python3
VENV_PATH := zen_venv
ZOO_GYM_PATH := ~/work/zoo/gym
HF_USERNAME := zenlm

# Model variants
MODELS := zen-nano-instruct zen-nano-instruct-4bit zen-nano-thinking zen-nano-thinking-4bit

# Default target
.PHONY: all
all: setup train quantize deploy

# =============================================================================
# Environment Setup
# =============================================================================

.PHONY: setup
setup: setup-venv setup-gym setup-tools
	@echo "✅ Complete environment setup finished"

setup-venv:
	@echo "🔧 Setting up Python virtual environment..."
	$(PYTHON) -m venv $(VENV_PATH)
	$(VENV_PATH)/bin/pip install --upgrade pip
	$(VENV_PATH)/bin/pip install mlx mlx-lm transformers datasets huggingface_hub torch
	$(VENV_PATH)/bin/pip install accelerate bitsandbytes peft
	@echo "✅ Virtual environment ready"

setup-gym:
	@echo "🏋️ Setting up zoo/gym training environment..."
	@if [ ! -d "$(ZOO_GYM_PATH)" ]; then \
		echo "📥 Cloning zoo/gym..."; \
		git clone https://github.com/zooai/gym $(ZOO_GYM_PATH); \
	fi
	@cd $(ZOO_GYM_PATH) && pip install -e .
	@echo "✅ Zoo/gym integration ready"

setup-tools:
	@echo "🔧 Setting up additional tools..."
	@if [ ! -d "llama.cpp" ]; then \
		echo "📥 Cloning llama.cpp for GGUF support..."; \
		git clone https://github.com/ggerganov/llama.cpp.git; \
		cd llama.cpp && make; \
	fi
	@echo "✅ Tools setup complete"

# =============================================================================
# Training Pipeline (using zoo/gym)
# =============================================================================

.PHONY: train
train: train-base train-instruct train-thinking
	@echo "✅ All model training complete"

train-base:
	@echo "🎯 Training base Zen-Nano model with zoo/gym..."
	cd $(ZOO_GYM_PATH) && python -m gym.train \
		--model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
		--dataset_name "zenlm/zen-identity" \
		--output_dir "../zen/zen-nano/models/zen-nano-4b-base" \
		--learning_rate 5e-5 \
		--num_train_epochs 3 \
		--per_device_train_batch_size 4 \
		--gradient_accumulation_steps 4 \
		--warmup_ratio 0.1 \
		--logging_steps 10 \
		--save_steps 500 \
		--max_seq_length 2048

train-instruct: train-base
	@echo "🎯 Fine-tuning for instruction following..."
	cd $(ZOO_GYM_PATH) && python -m gym.train \
		--model_name_or_path "../zen/zen-nano/models/zen-nano-4b-base" \
		--dataset_name "zenlm/zen-identity" \
		--output_dir "../zen/zen-nano/models/zen-nano-4b-instruct-base" \
		--learning_rate 2e-5 \
		--num_train_epochs 2 \
		--per_device_train_batch_size 8 \
		--instruction_template "User: {input}\\nAssistant: {output}"

train-thinking: train-base
	@echo "🧠 Training thinking variant with CoT..."
	cd $(ZOO_GYM_PATH) && python -m gym.train \
		--model_name_or_path "../zen/zen-nano/models/zen-nano-4b-base" \
		--dataset_name "zenlm/zen-identity" \
		--output_dir "../zen/zen-nano/models/zen-nano-4b-thinking-base" \
		--learning_rate 2e-5 \
		--num_train_epochs 2 \
		--per_device_train_batch_size 8 \
		--thinking_tokens \
		--instruction_template "User: {input}\\n<think>\\n{reasoning}\\n</think>\\nAssistant: {output}"

# =============================================================================
# Model Conversion and Quantization
# =============================================================================

.PHONY: quantize
quantize: convert-mlx convert-gguf quantize-4bit
	@echo "✅ All model quantization complete"

convert-mlx:
	@echo "🍎 Converting models to MLX format..."
	$(VENV_PATH)/bin/python -m mlx_lm.convert \
		--hf-path "./zen-nano/models/zen-nano-4b-instruct-base" \
		--mlx-path "./zen-nano/models/zen-nano-4b-instruct-mlx"
	$(VENV_PATH)/bin/python -m mlx_lm.convert \
		--hf-path "./zen-nano/models/zen-nano-4b-thinking-base" \
		--mlx-path "./zen-nano/models/zen-nano-4b-thinking-mlx"
	@echo "✅ MLX conversion complete"

convert-gguf:
	@echo "⚡ Converting models to GGUF format..."
	cd llama.cpp && python convert-hf-to-gguf.py \
		../zen-nano/models/zen-nano-4b-instruct-base \
		--outdir ../zen-nano/models/zen-nano-4b-instruct-gguf \
		--outtype f16
	cd llama.cpp && python convert-hf-to-gguf.py \
		../zen-nano/models/zen-nano-4b-thinking-base \
		--outdir ../zen-nano/models/zen-nano-4b-thinking-gguf \
		--outtype f16
	@echo "✅ GGUF conversion complete"

quantize-4bit:
	@echo "🔢 Creating 4-bit quantized versions..."
	$(VENV_PATH)/bin/python -m mlx_lm.quantize \
		--hf-path "./zen-nano/models/zen-nano-4b-instruct-mlx" \
		--q-bits 4 \
		--mlx-path "./zen-nano/models/zen-nano-4b-instruct-mlx-q4"
	$(VENV_PATH)/bin/python -m mlx_lm.quantize \
		--hf-path "./zen-nano/models/zen-nano-4b-thinking-mlx" \
		--q-bits 4 \
		--mlx-path "./zen-nano/models/zen-nano-4b-thinking-mlx-q4"
	@echo "✅ 4-bit quantization complete"

# =============================================================================
# Testing and Validation
# =============================================================================

.PHONY: test
test: test-identity test-performance test-formats
	@echo "✅ All tests completed"

test-identity:
	@echo "🧪 Testing model identity..."
	$(VENV_PATH)/bin/python test_zen_nano_identity.py

test-performance:
	@echo "📊 Running performance benchmarks..."
	$(VENV_PATH)/bin/python -m mlx_lm.benchmark --model ./zen-nano/models/zen-nano-4b-instruct-mlx

test-formats:
	@echo "🔧 Testing format compatibility..."
	$(VENV_PATH)/bin/python -c "from mlx_lm import load; load('./zen-nano/models/zen-nano-4b-instruct-mlx')"

# =============================================================================
# HuggingFace Deployment (Using Unified System)
# =============================================================================

.PHONY: deploy
deploy: test deploy-unified
	@echo "🚀 Complete deployment finished!"

deploy-unified:
	@echo "🚀 Deploying Zen models with unified system..."
	$(PYTHON) ../unified_deploy.py zen \
		--hf-username $(HF_USERNAME) \
		--base-path $(PWD)

deploy-dry-run:
	@echo "🔍 Dry run deployment (no upload)..."
	$(PYTHON) ../unified_deploy.py zen \
		--hf-username $(HF_USERNAME) \
		--base-path $(PWD) \
		--dry-run

# =============================================================================
# Development and Utilities
# =============================================================================

.PHONY: clean
clean:
	@echo "🧹 Cleaning up temporary files..."
	rm -rf __pycache__ .pytest_cache
	rm -f temp_*.md *.log
	find . -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

.PHONY: status
status:
	@echo "📊 Zen Model Ecosystem Status:"
	@echo "================================"
	@echo "MLX models: $(shell ls -d zen-nano/models/*-mlx* 2>/dev/null | wc -l)"
	@echo "HF repositories: 5 (4 models + 1 dataset)"
	@echo ""
	@echo "🔗 Live models:"
	@echo "• https://huggingface.co/$(HF_USERNAME)/zen-nano-instruct"
	@echo "• https://huggingface.co/$(HF_USERNAME)/zen-nano-instruct-4bit"
	@echo "• https://huggingface.co/$(HF_USERNAME)/zen-nano-thinking"
	@echo "• https://huggingface.co/$(HF_USERNAME)/zen-nano-thinking-4bit"
	@echo "• https://huggingface.co/datasets/$(HF_USERNAME)/zen-identity"

.PHONY: quick-test
quick-test:
	@echo "⚡ Quick model test..."
	$(VENV_PATH)/bin/python -c "\
from mlx_lm import load, generate; \
model, tokenizer = load('./zen-nano/models/zen-nano-4b-instruct-mlx'); \
response = generate(model, tokenizer, prompt='What is your name?', max_tokens=50); \
print('Response:', response)"

# =============================================================================
# Help and Documentation
# =============================================================================

.PHONY: help
help:
	@echo "Zen Model Ecosystem - Complete Reproduction Makefile"
	@echo "===================================================="
	@echo ""
	@echo "Main targets:"
	@echo "  all          - Complete pipeline: setup → train → quantize → deploy"
	@echo "  setup        - Set up environment, zoo/gym, and tools"
	@echo "  train        - Train all model variants using zoo/gym"
	@echo "  quantize     - Convert to MLX and GGUF formats + 4-bit quantization"
	@echo "  test         - Run identity, performance, and format tests"
	@echo "  deploy       - Deploy to HuggingFace with complete format support"
	@echo ""
	@echo "Development targets:"
	@echo "  status       - Show current ecosystem status"
	@echo "  clean        - Clean up temporary files"
	@echo "  quick-test   - Quick functionality test"
	@echo ""
	@echo "Examples:"
	@echo "  make setup              # Initial environment setup"
	@echo "  make train              # Train models with zoo/gym"
	@echo "  make deploy             # Deploy to HuggingFace"
	@echo "  make all                # Complete pipeline"
	@echo ""
	@echo "Requirements:"
	@echo "  - Python 3.8+"
	@echo "  - ~/work/zoo/gym (auto-cloned)"
	@echo "  - HF_TOKEN environment variable"
	@echo "  - 16GB+ RAM recommended"