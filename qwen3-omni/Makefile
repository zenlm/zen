# Qwen3-Omni Fine-tuning Makefile

.PHONY: help setup install clean train-unsloth train-instruct train-thinking test

help:
	@echo "Qwen3-Omni Fine-tuning Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup           - Complete setup with Unsloth"
	@echo "  make install         - Install dependencies"
	@echo "  make install-unsloth - Install Unsloth optimizations"
	@echo ""
	@echo "Training:"
	@echo "  make train-unsloth   - Train with Unsloth (fastest)"
	@echo "  make train-instruct  - Train instruction model"
	@echo "  make train-thinking  - Train reasoning model"
	@echo "  make train-captioner - Train audio captioning"
	@echo ""
	@echo "Data:"
	@echo "  make prepare-data    - Prepare multimodal datasets"
	@echo "  make download-model  - Download base models"
	@echo ""
	@echo "Inference:"
	@echo "  make test            - Test inference"
	@echo "  make serve           - Start API server"
	@echo ""
	@echo "Export:"
	@echo "  make export-gguf     - Export to GGUF format"
	@echo "  make export-16bit    - Export 16bit model"
	@echo "  make push-hub        - Push to HuggingFace"

# Environment setup
setup: install install-unsloth download-model
	@echo "✅ Setup complete!"

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

install-unsloth:
	@echo "Installing Unsloth with CUDA support..."
	pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
	pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
	@echo "✅ Unsloth installed"

# Model download
download-model:
	@echo "Downloading Qwen3-Omni models..."
	python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-Omni-30B-A3B-Instruct', trust_remote_code=True)"
	@echo "✅ Models downloaded"

download-model-thinking:
	@echo "Downloading Thinking model..."
	huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Thinking --local-dir ./models/thinking

download-model-captioner:
	@echo "Downloading Captioner model..."
	huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Captioner --local-dir ./models/captioner

# Data preparation
prepare-data:
	@echo "Preparing multimodal data..."
	python scripts/prepare_data.py \
		--output_dir data/processed \
		--format jsonl
	@echo "✅ Data prepared"

create-sample-data:
	@echo "Creating sample dataset..."
	python scripts/prepare_data.py \
		--output_dir data/sample \
		--create_sample
	@echo "✅ Sample data created"

# Training with Unsloth (fastest)
train-unsloth:
	@echo "Training with Unsloth optimizations..."
	python scripts/finetune_unsloth.py \
		--model qwen3-omni-instruct \
		--dataset data/multimodal/train.jsonl \
		--output checkpoints/unsloth \
		--epochs 3 \
		--batch_size 2 \
		--lora_r 128 \
		--save_16bit
	@echo "✅ Training complete"

# Train specific variants
train-instruct:
	@echo "Training Instruct model..."
	python scripts/finetune.py --config configs/instruct.yaml

train-thinking:
	@echo "Training Thinking model..."
	python scripts/finetune.py --config configs/thinking.yaml

train-captioner:
	@echo "Training Captioner model..."
	python scripts/finetune_unsloth.py \
		--model qwen3-omni-captioner \
		--dataset data/audio_captions/train.jsonl \
		--output checkpoints/captioner \
		--lora_r 64

# Quick training test
train-test:
	@echo "Running quick training test..."
	python scripts/finetune_unsloth.py \
		--model qwen3-omni-instruct \
		--dataset data/sample/sample.jsonl \
		--output checkpoints/test \
		--epochs 1 \
		--batch_size 1 \
		--lora_r 16
	@echo "✅ Test training complete"

# Inference testing
test:
	@echo "Testing inference..."
	python scripts/inference.py \
		--model checkpoints/unsloth \
		--prompt "What do you see and hear?" \
		--image data/test/image.jpg \
		--audio data/test/audio.wav

test-multimodal:
	@echo "Testing multimodal inference..."
	python scripts/inference.py \
		--model checkpoints/unsloth \
		--video data/test/video.mp4 \
		--use_audio_in_video

# Server
serve:
	@echo "Starting inference server..."
	python scripts/serve.py \
		--model checkpoints/unsloth \
		--port 8000

serve-vllm:
	@echo "Starting vLLM server..."
	python -m vllm.entrypoints.openai.api_server \
		--model checkpoints/unsloth_16bit \
		--port 8000 \
		--tensor-parallel-size 4

# Export models
export-gguf:
	@echo "Exporting to GGUF format..."
	python scripts/export.py \
		--model checkpoints/unsloth \
		--output exports/gguf \
		--quantization q4_k_m

export-16bit:
	@echo "Exporting 16bit model..."
	python scripts/export.py \
		--model checkpoints/unsloth \
		--output exports/16bit \
		--format merged_16bit

export-onnx:
	@echo "Exporting to ONNX..."
	python scripts/export.py \
		--model checkpoints/unsloth \
		--output exports/onnx \
		--format onnx

# Push to HuggingFace Hub
push-hub:
	@echo "Pushing to HuggingFace Hub..."
	python scripts/push_to_hub.py \
		--model checkpoints/unsloth \
		--repo $(HF_REPO) \
		--token $(HF_TOKEN)

# Benchmarking
benchmark:
	@echo "Running benchmarks..."
	python scripts/benchmark.py \
		--model checkpoints/unsloth \
		--benchmarks GSM8K,MMLU,HellaSwag

# Memory profiling
profile:
	@echo "Profiling memory usage..."
	python scripts/profile_memory.py \
		--model qwen3-omni-instruct \
		--batch_size 1 \
		--seq_length 32768

# Cleanup
clean:
	@echo "Cleaning up..."
	rm -rf checkpoints/test
	rm -rf exports/temp
	rm -rf __pycache__ */__pycache__
	rm -rf .cache
	@echo "✅ Cleanup complete"

clean-all: clean
	@echo "Deep cleaning..."
	rm -rf checkpoints/*
	rm -rf exports/*
	rm -rf data/processed/*
	rm -rf wandb/
	rm -rf logs/
	@echo "✅ Deep clean complete"

# Docker support
docker-build:
	@echo "Building Docker image..."
	docker build -t qwen3-omni:latest .

docker-run:
	docker run --gpus all -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/checkpoints:/app/checkpoints \
		qwen3-omni:latest

# Development helpers
format:
	@echo "Formatting code..."
	black scripts/ --line-length 100
	isort scripts/

lint:
	@echo "Linting code..."
	flake8 scripts/ --max-line-length 100
	mypy scripts/

# Monitor training
monitor:
	@echo "Opening monitoring tools..."
	@echo "Tensorboard: http://localhost:6006"
	@echo "Wandb: https://wandb.ai/$(USER)/qwen3-omni-finetuning"
	tensorboard --logdir logs/

# GPU monitoring
gpu-status:
	@watch -n 1 nvidia-smi

# Requirements management
freeze:
	pip freeze > requirements.lock

update-deps:
	pip install --upgrade -r requirements.txt