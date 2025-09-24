# Makefile for ZenLM fine-tuning

VENV_DIR := zen_venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
ADAPTER_DIR := adapters

.PHONY: all clean setup train test

all: train test

setup: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(PIP) install -r requirements.txt
	touch $(VENV_DIR)/bin/activate

train: $(ADAPTER_DIR)/adapters.safetensors

$(ADAPTER_DIR)/adapters.safetensors: setup data/train.jsonl
	$(PYTHON) -m mlx_lm_lora.train --train --model mlx-community/Qwen3-4B-Instruct-2507-4bit --data data --adapter-path $(ADAPTER_DIR) --iters 100 --batch-size 2

data/train.jsonl: prepare_data.py training_data.py
	$(PYTHON) prepare_data.py

test: setup $(ADAPTER_DIR)/adapters.safetensors generate.py
	$(PYTHON) generate.py

clean:
	rm -rf $(VENV_DIR) data $(ADAPTER_DIR) mlx_lm_lora
	rm -f models.py requirements.txt generate.py
	rm -rf mlx-lm-lora