# Makefile for Zen1-Omni Project

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
HF_TOKEN := $(shell cat ~/.huggingface/token)
BRAND_NAME ?= zenml
DATASET_NAME := $(BRAND_NAME)/zen1-omni-preferences
OUTPUT_DIR := $(BRAND_NAME)/zen1-omni

# Targets
.PHONY: help setup clean install-dependencies upload-dataset train-gspo train-autotrain deploy-spaces evaluate-model full-pipeline

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help                  Show this help message."
	@echo "  setup                 Install dependencies and setup the environment."
	@echo "  clean                 Remove temporary files and build artifacts."
	@echo "  install-dependencies  Install Python and system dependencies."
	@echo "  upload-dataset        Upload the dataset to HuggingFace Hub."
	@echo "  train-gspo            Train the model with GSPO."
	@echo "  train-autotrain       Train the model with AutoTrain."
	@echo "  deploy-spaces         Deploy the model to HuggingFace Spaces."
	@echo "  evaluate-model        Evaluate the trained model."
	@echo "  full-pipeline         Run the full pipeline: setup, upload, train, evaluate, deploy."

setup: install-dependencies
	@echo "Environment setup complete."

clean:
	@echo "Cleaning up project..."
	@rm -rf zen1-omni-demo
	@rm -rf eval_results
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@echo "Cleanup complete."

install-dependencies:
	@echo "Installing dependencies..."
	$(PIP) install -r training/requirements.txt
	$(PIP) install -U "huggingface_hub[cli]"
	$(PIP) install -U flash-attn --no-build-isolation
	$(PIP) install git+https://github.com/huggingface/transformers
	$(PIP) install accelerate
	$(PIP) install qwen-omni-utils -U
	@echo "Dependencies installed."

upload-dataset:
	@echo "Uploading dataset..."
	@bash -c "source ~/.bashrc && ./training/launch_training.sh upload"
	@echo "Dataset uploaded."

train-gspo:
	@echo "Starting GSPO training..."
	@bash -c "source ~/.bashrc && ./training/launch_training.sh gspo"
	@echo "GSPO training complete."

train-autotrain:
	@echo "Starting AutoTrain..."
	@bash -c "source ~/.bashrc && ./training/launch_training.sh autotrain"
	@echo "AutoTrain complete."

deploy-spaces:
	@echo "Deploying to HuggingFace Spaces..."
	@bash -c "source ~/.bashrc && ./training/launch_training.sh deploy"
	@echo "Deployment to Spaces complete."

evaluate-model:
	@echo "Running evaluation..."
	@bash -c "source ~/.bashrc && ./training/launch_training.sh evaluate"
	@echo "Evaluation complete."

full-pipeline: setup upload-dataset train-gspo evaluate-model deploy-spaces
	@echo "Full pipeline executed successfully."

