# --- Variables ---
MAKEFILE_DIR     := $(dir $(lastword $(MAKEFILE_LIST)))
MODEL_NAME       := zen-nano
MODEL_BASE       := mlx-community/Qwen3-4B-Instruct-2507-4bit # Base model for training
DATA_DIR         := $(MAKEFILE_DIR)data
TRAIN_SCRIPT     := $(MAKEFILE_DIR)train.py
OUTPUT_DIR       := $(MAKEFILE_DIR)model
PAPER_DIR        := $(MAKEFILE_DIR)paper
VENV_DIR         := $(MAKEFILE_DIR)../zen_venv
PYTHON           := $(VENV_DIR)/bin/python
PIP              := $(VENV_DIR)/bin/pip

.PHONY: all install data train package deploy paper clean

all: install data train package

# --- Installation ---
install: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: $(MAKEFILE_DIR)../requirements_finetune.txt
	test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(PIP) install -r $(MAKEFILE_DIR)../requirements_finetune.txt
	touch $(VENV_DIR)/bin/activate

# --- Data ---
data: $(DATA_DIR)/train.jsonl

$(DATA_DIR)/train.jsonl:
	@echo "Preparing training data..."
	# Placeholder for data preparation script
	# You would run your data preparation script here.
	# For now, we assume the data is already in place.
	@if [ ! -f "$(DATA_DIR)/train.jsonl" ]; then \
		echo "Error: $(DATA_DIR)/train.jsonl not found. Please prepare your training data."; \
		exit 1; \
	fi

# --- Training & Fusing ---
train: $(OUTPUT_DIR)/fused/config.json

$(OUTPUT_DIR)/fused/config.json: $(TRAIN_SCRIPT) $(DATA_DIR)/train.jsonl install
	@echo "Starting model training and fusing..."
	$(PYTHON) $(TRAIN_SCRIPT) \
		--model $(MODEL_BASE) \
		--data $(DATA_DIR) \
		--adapter-path $(OUTPUT_DIR) \
		--fuse

# --- Packaging ---
package: gguf
	@echo "Packaging model..."

gguf: $(MAKEFILE_DIR)zen-nano.gguf

$(MAKEFILE_DIR)zen-nano.gguf: add_architecture
	@echo "Converting model to GGUF..."
	$(PYTHON) $(MAKEFILE_DIR)convert_to_gguf.py

add_architecture: $(OUTPUT_DIR)/fused/config.json
	@echo "Adding architecture to config.json..."
	$(PYTHON) $(MAKEFILE_DIR)add_architecture.py

# --- Deployment ---
deploy:
	@echo "Deploying model..."
	# Placeholder for deployment script (e.g., pushing to Hugging Face)

# --- Paper ---
paper:
	@echo "Compiling LaTeX paper..."
	# Placeholder for LaTeX compilation command
	# (cd $(PAPER_DIR) && pdflatex main.tex)

# --- Clean ---
clean:
	@echo "Cleaning up..."
	rm -rf $(OUTPUT_DIR)
	rm -f $(DATA_DIR)/*.jsonl
