# Zen Nano Training Pipeline - Now with Gym Integration! 🏋️
# Jointly developed by Hanzo AI Inc and Zoo Labs Foundation
# Version: 2.0 - Integrated with Gym Platform

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Configuration
MODEL_NAME = Zen Nano v1.0
BASE_MODEL_QWEN3 = Qwen/Qwen3-4B-Instruct
BASE_MODEL_MLX = ../base-models/Qwen3-4B-Instruct-2507
TRAINING_DATA = training/zen_nano_clean.jsonl

# Path Configuration
ADAPTER_PATH = models/adapters
FUSED_PATH = models/fused
SCRIPTS_DIR = scripts
GYM_PATH = /Users/z/work/zoo/gym
GYM_OUTPUT = $(GYM_PATH)/output/zen-nano
PYTHON = python3.12

# Training Configuration
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
LORA_RANK = 16

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[0;33m
CYAN = \033[0;36m
RED = \033[0;31m
BOLD = \033[1m
NC = \033[0m # No Color

# ============================================================================
# HELP & DEFAULT TARGET
# ============================================================================

.PHONY: help
help:
	@echo "$(BOLD)$(CYAN)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BOLD)$(CYAN)║     🧘 Zen Nano Training Pipeline v2.0                     ║$(NC)"
	@echo "$(BOLD)$(CYAN)║     Now Powered by Gym Platform! 🏋️                        ║$(NC)"
	@echo "$(BOLD)$(CYAN)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(BOLD)$(GREEN)Jointly developed by:$(NC)"
	@echo "  • Hanzo AI Inc (Techstars LA)"
	@echo "  • Zoo Labs Foundation (501c3 SF)"
	@echo ""
	@echo "$(BOLD)$(YELLOW)🚀 QUICK START:$(NC)"
	@echo "  $(GREEN)make gym-ui$(NC)        - Launch visual training interface (recommended)"
	@echo "  $(GREEN)make gym-train$(NC)     - Train with Gym CLI (fast)"
	@echo "  $(GREEN)make gym-test$(NC)      - Test your trained model"
	@echo ""
	@echo "$(BOLD)$(YELLOW)📋 GYM COMMANDS (New!):$(NC)"
	@echo "  $(CYAN)make gym-setup$(NC)     - Setup Gym integration (run once)"
	@echo "  $(CYAN)make gym-ui$(NC)        - Launch Gym Web UI for training"
	@echo "  $(CYAN)make gym-train$(NC)     - Train using Gym CLI"
	@echo "  $(CYAN)make gym-qlora$(NC)     - Train with QLoRA (low memory)"
	@echo "  $(CYAN)make gym-test$(NC)      - Test model with chat interface"
	@echo "  $(CYAN)make gym-export$(NC)    - Export to GGUF for deployment"
	@echo "  $(CYAN)make gym-serve$(NC)     - Serve model as API"
	@echo "  $(CYAN)make gym-status$(NC)    - Check training status"
	@echo ""
	@echo "$(BOLD)$(YELLOW)🔧 MLX COMMANDS (Original):$(NC)"
	@echo "  $(CYAN)make mlx-train$(NC)     - Train with MLX (Apple Silicon)"
	@echo "  $(CYAN)make mlx-test$(NC)      - Test MLX model"
	@echo "  $(CYAN)make mlx-fuse$(NC)      - Fuse MLX adapters"
	@echo ""
	@echo "$(BOLD)$(YELLOW)🛠️ UTILITIES:$(NC)"
	@echo "  $(CYAN)make prepare$(NC)       - Prepare training data"
	@echo "  $(CYAN)make clean$(NC)         - Clean generated files"
	@echo "  $(CYAN)make validate$(NC)      - Validate training data"
	@echo "  $(CYAN)make benchmark$(NC)     - Run performance benchmarks"
	@echo "  $(CYAN)make info$(NC)          - Show project information"
	@echo "  $(CYAN)make all$(NC)           - Run complete pipeline"
	@echo ""
	@echo "$(BOLD)$(YELLOW)📦 DEPLOYMENT:$(NC)"
	@echo "  $(CYAN)make deploy-ollama$(NC) - Deploy to Ollama"
	@echo "  $(CYAN)make deploy-hf$(NC)     - Deploy to Hugging Face"
	@echo "  $(CYAN)make deploy-gguf$(NC)   - Create GGUF for llama.cpp"
	@echo ""
	@echo "$(BOLD)For detailed documentation, see README.md$(NC)"

# ============================================================================
# GYM INTEGRATION COMMANDS 🏋️
# ============================================================================

.PHONY: gym-setup
gym-setup:
	@echo "$(BOLD)$(CYAN)🔧 Setting up Gym integration...$(NC)"
	@$(PYTHON) setup_gym_integration.py
	@echo "$(GREEN)✅ Gym integration ready!$(NC)"

.PHONY: gym-ui
gym-ui: gym-setup
	@echo "$(BOLD)$(CYAN)🌐 Launching Gym Web UI...$(NC)"
	@echo ""
	@echo "$(YELLOW)📝 Instructions:$(NC)"
	@echo "  1. UI will open at http://localhost:7860"
	@echo "  2. Select Model: $(GREEN)Qwen/Qwen3-4B-Instruct$(NC)"
	@echo "  3. Select Dataset: $(GREEN)zen_nano$(NC)"
	@echo "  4. Select Method: $(GREEN)QLoRA$(NC)"
	@echo "  5. Click Start Training!"
	@echo ""
	@cd $(GYM_PATH) && $(PYTHON) -m llamafactory.webui.interface

.PHONY: gym-train
gym-train: gym-setup
	@echo "$(BOLD)$(CYAN)🚀 Training with Gym CLI...$(NC)"
	@echo "Model: $(BASE_MODEL_QWEN3)"
	@echo "Dataset: zen_nano"
	@echo "Method: QLoRA (4-bit)"
	@echo "Epochs: $(EPOCHS)"
	@echo ""
	@cd $(GYM_PATH) && $(PYTHON) -m llamafactory.train \
		--model_name_or_path "$(BASE_MODEL_QWEN3)" \
		--template "qwen3" \
		--dataset "zen_nano" \
		--dataset_dir "./data" \
		--cutoff_len 2048 \
		--stage "sft" \
		--do_train True \
		--finetuning_type "lora" \
		--quantization_bit 4 \
		--lora_rank $(LORA_RANK) \
		--lora_alpha 32 \
		--lora_dropout 0.05 \
		--lora_target "all" \
		--per_device_train_batch_size $(BATCH_SIZE) \
		--gradient_accumulation_steps 8 \
		--learning_rate $(LEARNING_RATE) \
		--num_train_epochs $(EPOCHS) \
		--lr_scheduler_type "cosine" \
		--warmup_ratio 0.1 \
		--gradient_checkpointing True \
		--output_dir "./output/zen-nano" \
		--logging_steps 10 \
		--save_steps 100 \
		--save_total_limit 3 \
		--plot_loss True \
		--overwrite_output_dir True
	@echo "$(GREEN)✅ Training complete! Model saved to $(GYM_OUTPUT)$(NC)"

.PHONY: gym-qlora
gym-qlora: gym-train
	@echo "$(GREEN)✅ QLoRA training complete!$(NC)"

.PHONY: gym-test
gym-test:
	@echo "$(BOLD)$(CYAN)🧪 Testing Zen Nano model...$(NC)"
	@echo "Type 'exit' to quit the chat"
	@echo ""
	@cd $(GYM_PATH) && $(PYTHON) -m llamafactory.chat \
		--model_name_or_path "$(BASE_MODEL_QWEN3)" \
		--adapter_name_or_path "./output/zen-nano" \
		--template "qwen3"

.PHONY: gym-export
gym-export:
	@echo "$(BOLD)$(CYAN)📦 Exporting to GGUF format...$(NC)"
	@cd $(GYM_PATH) && $(PYTHON) -m llamafactory.export \
		--model_name_or_path "$(BASE_MODEL_QWEN3)" \
		--adapter_name_or_path "./output/zen-nano" \
		--template "qwen3" \
		--export_dir "./exports" \
		--export_size 4 \
		--export_quantization_bit 4
	@echo "$(GREEN)✅ Exported to $(GYM_PATH)/exports/$(NC)"

.PHONY: gym-serve
gym-serve:
	@echo "$(BOLD)$(CYAN)🌐 Serving model as API...$(NC)"
	@echo "API will be available at http://localhost:8000"
	@cd $(GYM_PATH) && $(PYTHON) -m llamafactory.api \
		--model_name_or_path "$(BASE_MODEL_QWEN3)" \
		--adapter_name_or_path "./output/zen-nano" \
		--template "qwen3" \
		--port 8000

.PHONY: gym-status
gym-status:
	@echo "$(BOLD)$(CYAN)📊 Checking training status...$(NC)"
	@if [ -d "$(GYM_OUTPUT)" ]; then \
		echo "$(GREEN)✅ Model found at: $(GYM_OUTPUT)$(NC)"; \
		ls -lah $(GYM_OUTPUT)/*.safetensors 2>/dev/null || echo "No model files yet"; \
		echo ""; \
		echo "Latest checkpoint:"; \
		ls -lt $(GYM_OUTPUT)/checkpoint-* 2>/dev/null | head -1 || echo "No checkpoints yet"; \
	else \
		echo "$(YELLOW)⚠️  No training output found. Run 'make gym-train' first$(NC)"; \
	fi

# ============================================================================
# MLX COMMANDS (Original)
# ============================================================================

.PHONY: mlx-train
mlx-train: prepare
	@echo "$(CYAN)🚀 Starting MLX LoRA finetuning...$(NC)"
	@$(PYTHON) -m mlx_lm lora \
		--model $(BASE_MODEL_MLX) \
		--data training \
		--train \
		--batch-size $(BATCH_SIZE) \
		--num-layers 16 \
		--iters 200 \
		--learning-rate 5e-5 \
		--adapter-path $(ADAPTER_PATH)
	@echo "$(GREEN)✅ MLX training complete$(NC)"

.PHONY: mlx-test
mlx-test:
	@echo "$(CYAN)🧪 Testing MLX model...$(NC)"
	@$(PYTHON) $(SCRIPTS_DIR)/test_identity.py

.PHONY: mlx-fuse
mlx-fuse:
	@echo "$(CYAN)🔗 Fusing MLX adapters...$(NC)"
	@$(PYTHON) -m mlx_lm fuse \
		--model $(BASE_MODEL_MLX) \
		--adapter-path $(ADAPTER_PATH) \
		--save-path $(FUSED_PATH)

# ============================================================================
# DATA PREPARATION
# ============================================================================

.PHONY: prepare
prepare:
	@echo "$(CYAN)📝 Preparing training data...$(NC)"
	@$(PYTHON) $(SCRIPTS_DIR)/prepare_data.py 2>/dev/null || echo "Prepare script not found, using existing data"
	@echo "Training data: $(TRAINING_DATA)"
	@wc -l $(TRAINING_DATA)
	@echo "$(GREEN)✅ Data ready$(NC)"

.PHONY: validate
validate:
	@echo "$(CYAN)🔍 Validating training data...$(NC)"
	@$(PYTHON) -c "import json; [json.loads(line) for line in open('$(TRAINING_DATA)')]"
	@echo "$(GREEN)✅ Training data valid$(NC)"

# ============================================================================
# DEPLOYMENT
# ============================================================================

.PHONY: deploy-ollama
deploy-ollama: gym-export
	@echo "$(CYAN)🚢 Deploying to Ollama...$(NC)"
	@echo "FROM $(GYM_PATH)/exports/zen-nano.gguf" > Modelfile.zen-nano
	@echo "SYSTEM \"You are Zen Nano v1.0, jointly developed by Hanzo AI Inc and Zoo Labs Foundation.\"" >> Modelfile.zen-nano
	@ollama create zen-nano -f Modelfile.zen-nano
	@echo "$(GREEN)✅ Deployed as 'zen-nano' in Ollama$(NC)"
	@echo "Run with: ollama run zen-nano"

.PHONY: deploy-hf
deploy-hf:
	@echo "$(CYAN)🤗 Deploying to Hugging Face...$(NC)"
	@cd $(GYM_PATH) && $(PYTHON) scripts/upload_to_hub.py \
		--model_name_or_path "$(BASE_MODEL_QWEN3)" \
		--adapter_name_or_path "./output/zen-nano" \
		--hub_model_id "zooai/zen-nano-4b"
	@echo "$(GREEN)✅ Deployed to HuggingFace$(NC)"

.PHONY: deploy-gguf
deploy-gguf: gym-export
	@echo "$(GREEN)✅ GGUF file ready at $(GYM_PATH)/exports/$(NC)"

# ============================================================================
# UTILITIES
# ============================================================================

.PHONY: clean
clean:
	@echo "$(CYAN)🧹 Cleaning generated files...$(NC)"
	@rm -rf $(ADAPTER_PATH) $(FUSED_PATH) $(GYM_OUTPUT)
	@rm -f training/*.json Modelfile.zen-nano
	@echo "$(GREEN)✅ Cleaned$(NC)"

.PHONY: info
info:
	@echo "$(BOLD)$(CYAN)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BOLD)$(CYAN)║                  Zen Nano Information                      ║$(NC)"
	@echo "$(BOLD)$(CYAN)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo "Version:       v1.0"
	@echo "Model:         Qwen3-4B base"
	@echo "Parameters:    4B (quantized to 2-3GB)"
	@echo "Training:      Gym Platform (zoogym)"
	@echo ""
	@echo "$(BOLD)Creators:$(NC)"
	@echo "  • Hanzo AI Inc (Los Angeles, Techstars '24)"
	@echo "  • Zoo Labs Foundation (San Francisco, 501c3)"
	@echo ""
	@echo "$(BOLD)Features:$(NC)"
	@echo "  ✅ Edge deployment ready"
	@echo "  ✅ Offline capable"
	@echo "  ✅ Privacy preserving"
	@echo "  ✅ Low carbon footprint"
	@echo "  ✅ Open source (Apache 2.0)"
	@echo ""
	@echo "$(BOLD)Training Data:$(NC)"
	@if [ -f "$(TRAINING_DATA)" ]; then \
		echo "  • File: $(TRAINING_DATA)"; \
		echo "  • Examples: $$(wc -l < $(TRAINING_DATA))"; \
	else \
		echo "  • No training data found"; \
	fi

.PHONY: benchmark
benchmark:
	@echo "$(CYAN)⚡ Running performance benchmarks...$(NC)"
	@echo "Testing inference speed..."
	@cd $(GYM_PATH) && time $(PYTHON) -c "from llamafactory.chat import ChatModel; \
		model = ChatModel(dict(model_name_or_path='$(BASE_MODEL_QWEN3)', \
		adapter_name_or_path='./output/zen-nano', template='qwen3')); \
		print(model.chat('Hello')[0])"

# ============================================================================
# COMBINED WORKFLOWS
# ============================================================================

.PHONY: all
all: gym-setup gym-train gym-test
	@echo "$(BOLD)$(GREEN)✅ Complete pipeline finished!$(NC)"
	@echo "Model trained and tested successfully"

.PHONY: quick
quick: gym-setup
	@echo "$(BOLD)$(YELLOW)⚡ Quick training (1 epoch, small batch)$(NC)"
	@$(MAKE) gym-train EPOCHS=1 BATCH_SIZE=1

# ============================================================================
# ADVANCED OPTIONS
# ============================================================================

.PHONY: tensorboard
tensorboard:
	@echo "$(CYAN)📊 Launching TensorBoard...$(NC)"
	@cd $(GYM_PATH) && tensorboard --logdir ./output/zen-nano/runs

.PHONY: monitor
monitor:
	@echo "$(CYAN)📈 Monitoring training progress...$(NC)"
	@while true; do \
		clear; \
		echo "$(BOLD)Training Progress$(NC)"; \
		echo "================"; \
		tail -20 $(GYM_OUTPUT)/trainer_log.jsonl 2>/dev/null || echo "No logs yet"; \
		sleep 5; \
	done

.PHONY: compare
compare: gym-test mlx-test
	@echo "$(CYAN)📊 Comparing Gym vs MLX models...$(NC)"
	@echo "Both models tested. Check outputs above."

# Default target
.DEFAULT_GOAL := help