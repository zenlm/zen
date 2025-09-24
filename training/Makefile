# Zen Model Training Pipeline
# Unified commands for training all model variants

.PHONY: help setup prepare-data train test clean

# Default target
help:
	@echo "Zen Model Training Pipeline"
	@echo "=========================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup              - Install dependencies and setup environment"
	@echo "  make prepare-data       - Prepare all training datasets"
	@echo ""
	@echo "Training Commands:"
	@echo "  make train-nano         - Train zen-nano base model"
	@echo "  make train-nano-instruct - Train zen-nano instruction model"
	@echo "  make train-nano-thinking - Train zen-nano thinking model"
	@echo "  make train-omni         - Train zen-omni multimodal model"
	@echo "  make train-coder        - Train zen-coder specialized model"
	@echo "  make train-next         - Train zen-next experimental model"
	@echo ""
	@echo "Quick Training (LoRA):"
	@echo "  make quick-nano         - Quick train zen-nano with LoRA"
	@echo "  make quick-omni         - Quick train zen-omni with QLoRA"
	@echo ""
	@echo "BitDelta Training:"
	@echo "  make bitdelta-nano      - Train zen-nano with BitDelta"
	@echo "  make bitdelta-next      - Train zen-next with BitDelta"
	@echo ""
	@echo "Utilities:"
	@echo "  make test               - Test training pipeline"
	@echo "  make clean              - Clean checkpoints and logs"
	@echo "  make monitor            - Monitor training progress"
	@echo ""

# Setup environment
setup:
	@echo "Setting up Zen training environment..."
	pip install -r requirements.txt
	@echo "Environment setup complete!"

# Prepare datasets
prepare-data:
	@echo "Preparing training datasets..."
	python prepare_datasets.py --output-dir ./data
	@echo "Datasets prepared!"

# Data preparation for specific stages
prepare-instruct:
	python prepare_datasets.py --output-dir ./data --stage instruct

prepare-thinking:
	python prepare_datasets.py --output-dir ./data --stage thinking

prepare-code:
	python prepare_datasets.py --output-dir ./data --include-repos ../lux ../zoo

# === ZEN-NANO MODELS ===

train-nano:
	@echo "Training zen-nano base model..."
	python train.py \
		--model zen-nano \
		--stage base \
		--epochs 3 \
		--batch-size 4 \
		--output-dir ./checkpoints/zen-nano

train-nano-instruct:
	@echo "Training zen-nano instruction model..."
	python train.py \
		--model zen-nano-instruct \
		--stage instruct \
		--epochs 3 \
		--batch-size 4 \
		--output-dir ./checkpoints/zen-nano-instruct

train-nano-thinking:
	@echo "Training zen-nano thinking model..."
	python train.py \
		--model zen-nano-thinking \
		--stage thinking \
		--epochs 3 \
		--batch-size 2 \
		--output-dir ./checkpoints/zen-nano-thinking

# Quick LoRA training for zen-nano
quick-nano:
	@echo "Quick training zen-nano with LoRA..."
	python train.py \
		--model zen-nano \
		--stage instruct \
		--use-lora \
		--epochs 1 \
		--batch-size 8 \
		--output-dir ./checkpoints/zen-nano-lora

# === ZEN-OMNI MODELS ===

train-omni:
	@echo "Training zen-omni multimodal model..."
	python train.py \
		--model zen-omni \
		--stage instruct \
		--epochs 3 \
		--batch-size 1 \
		--output-dir ./checkpoints/zen-omni

train-omni-instruct:
	@echo "Training zen-omni instruction model..."
	python train.py \
		--model zen-omni-instruct \
		--stage instruct \
		--epochs 3 \
		--batch-size 1 \
		--output-dir ./checkpoints/zen-omni-instruct

train-omni-thinking:
	@echo "Training zen-omni thinking model..."
	python train.py \
		--model zen-omni-thinking \
		--stage thinking \
		--epochs 3 \
		--batch-size 1 \
		--output-dir ./checkpoints/zen-omni-thinking

train-omni-captioner:
	@echo "Training zen-omni captioner model..."
	python train.py \
		--model zen-omni-captioner \
		--stage instruct \
		--epochs 3 \
		--batch-size 2 \
		--output-dir ./checkpoints/zen-omni-captioner

# Quick QLoRA training for zen-omni
quick-omni:
	@echo "Quick training zen-omni with QLoRA..."
	python train.py \
		--model zen-omni \
		--stage instruct \
		--use-qlora \
		--epochs 1 \
		--batch-size 1 \
		--output-dir ./checkpoints/zen-omni-qlora

# === SPECIALIZED MODELS ===

train-coder:
	@echo "Training zen-coder model..."
	python train.py \
		--model zen-coder \
		--stage instruct \
		--epochs 3 \
		--batch-size 4 \
		--output-dir ./checkpoints/zen-coder

train-next:
	@echo "Training zen-next experimental model..."
	python train.py \
		--model zen-next \
		--stage instruct \
		--epochs 3 \
		--batch-size 4 \
		--output-dir ./checkpoints/zen-next

# === BITDELTA TRAINING ===

bitdelta-nano:
	@echo "Training zen-nano with BitDelta..."
	python train.py \
		--model zen-nano \
		--stage instruct \
		--use-bitdelta \
		--epochs 3 \
		--batch-size 4 \
		--output-dir ./checkpoints/zen-nano-bitdelta

bitdelta-next:
	@echo "Training zen-next with BitDelta..."
	python train.py \
		--model zen-next \
		--stage instruct \
		--use-bitdelta \
		--epochs 3 \
		--batch-size 4 \
		--output-dir ./checkpoints/zen-next-bitdelta

# === DISTRIBUTED TRAINING ===

train-omni-distributed:
	@echo "Training zen-omni with distributed setup..."
	torchrun --nproc_per_node=4 train.py \
		--model zen-omni \
		--stage instruct \
		--epochs 3 \
		--batch-size 1 \
		--output-dir ./checkpoints/zen-omni-distributed

# === TESTING ===

test:
	@echo "Testing training pipeline..."
	python train.py \
		--model zen-nano \
		--stage instruct \
		--use-lora \
		--epochs 1 \
		--batch-size 2 \
		--output-dir ./test_checkpoint
	@echo "Training test completed!"

test-bitdelta:
	@echo "Testing BitDelta integration..."
	python -c "from utils.bitdelta_integration import ZenBitDelta; print('BitDelta integration working!')"

# === MONITORING ===

monitor:
	@echo "Starting training monitor..."
	tensorboard --logdir ./checkpoints/*/logs

# === CONVERSION ===

convert-gguf:
	@echo "Converting models to GGUF format..."
	@for model in zen-nano zen-omni zen-coder; do \
		if [ -d "./checkpoints/$$model" ]; then \
			python ../gguf-conversion/convert_zen_to_gguf.py \
				--model ./checkpoints/$$model \
				--output ./gguf/$$model.gguf; \
		fi \
	done

convert-mlx:
	@echo "Converting models to MLX format..."
	@for model in zen-nano zen-omni; do \
		if [ -d "./checkpoints/$$model" ]; then \
			python ../mlx-conversion/convert.py \
				--model ./checkpoints/$$model \
				--output ./mlx/$$model; \
		fi \
	done

# === UTILITIES ===

clean:
	@echo "Cleaning checkpoints and logs..."
	rm -rf ./checkpoints/test_checkpoint
	rm -rf ./logs/test_*
	find ./checkpoints -name "checkpoint-*" -type d | sort -r | tail -n +4 | xargs rm -rf
	@echo "Cleanup complete!"

clean-all:
	@echo "Cleaning all training artifacts..."
	rm -rf ./checkpoints
	rm -rf ./logs
	rm -rf ./data/*.jsonl
	@echo "All artifacts cleaned!"

# === DEPLOYMENT ===

deploy-ollama:
	@echo "Deploying models to Ollama..."
	@for model in zen-nano zen-omni zen-coder; do \
		if [ -f "./gguf/$$model.gguf" ]; then \
			ollama create hanzo/$$model -f ../Modelfile.$$model; \
		fi \
	done

push-hub:
	@echo "Pushing models to HuggingFace Hub..."
	@for model in zen-nano zen-omni zen-coder; do \
		if [ -d "./checkpoints/$$model" ]; then \
			python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
				model = AutoModelForCausalLM.from_pretrained('./checkpoints/$$model'); \
				tokenizer = AutoTokenizer.from_pretrained('./checkpoints/$$model'); \
				model.push_to_hub('hanzo-ai/$$model'); \
				tokenizer.push_to_hub('hanzo-ai/$$model')"; \
		fi \
	done

# === BENCHMARKS ===

benchmark:
	@echo "Running benchmarks..."
	python benchmark.py --models zen-nano zen-omni zen-coder

# === DOCUMENTATION ===

docs:
	@echo "Generating training documentation..."
	@echo "# Zen Model Training Results\n" > TRAINING_RESULTS.md
	@echo "## Models Trained\n" >> TRAINING_RESULTS.md
	@for model in ./checkpoints/*/; do \
		if [ -f "$$model/training_summary.json" ]; then \
			echo "### $$(basename $$model)" >> TRAINING_RESULTS.md; \
			cat "$$model/README.md" >> TRAINING_RESULTS.md; \
			echo "\n---\n" >> TRAINING_RESULTS.md; \
		fi \
	done
	@echo "Documentation generated: TRAINING_RESULTS.md"