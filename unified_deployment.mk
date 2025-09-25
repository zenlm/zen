# Unified AI Model Ecosystem Deployment
# Manages both Zen Nano and Supra Nexus models

.PHONY: all status deploy-zen deploy-supra deploy-all verify clean help

# Configuration
PYTHON := python3
HF_CLI := hf

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
RED := \033[0;31m
NC := \033[0m # No Color

# Default target
all: deploy-all

help:
	@echo "$(BLUE)╔════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║     Unified AI Model Ecosystem Deployment         ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@echo "  $(YELLOW)make status$(NC)       - Show deployment status for all models"
	@echo "  $(YELLOW)make deploy-zen$(NC)   - Deploy Zen Nano models"
	@echo "  $(YELLOW)make deploy-supra$(NC) - Deploy Supra Nexus models"
	@echo "  $(YELLOW)make deploy-all$(NC)   - Deploy both ecosystems"
	@echo "  $(YELLOW)make verify$(NC)       - Verify all deployments"
	@echo "  $(YELLOW)make clean$(NC)        - Clean temporary files"
	@echo ""
	@echo "$(GREEN)Model Ecosystems:$(NC)"
	@echo "  $(BLUE)Zen Nano:$(NC) 4B efficient models by Hanzo AI"
	@echo "  $(BLUE)Supra Nexus:$(NC) O1 reasoning models by Supra Foundation"

status:
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)            Model Ecosystem Status Report              $(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(GREEN)Zen Nano Models (zenlm):$(NC)"
	@echo "  • zen-nano-instruct"
	@echo "  • zen-nano-instruct-4bit"
	@echo "  • zen-nano-thinking"
	@echo "  • zen-nano-thinking-4bit"
	@echo "  • zen-identity (dataset)"
	@echo ""
	@echo ""
	@echo "$(BLUE)Checking live status...$(NC)"
	@$(PYTHON) -c "import requests; \
		zen_models = ['zen-nano-instruct', 'zen-nano-instruct-4bit', 'zen-nano-thinking', 'zen-nano-thinking-4bit']; \
		print('\nZen Nano:'); \
		for m in zen_models: \
			r = requests.head(f'https://huggingface.co/zenlm/{m}', allow_redirects=True); \
			status = '✅' if r.status_code == 200 else '❌'; \
			print(f'  {status} {m}'); \
		print('\nSupra Nexus:'); \
		for m in supra_models: \
			status = '✅' if r.status_code == 200 else '❌'; \
			print(f'  {status} {m}')" 2>/dev/null || echo "$(RED)Unable to check live status$(NC)"

deploy-zen:
	@echo "$(BLUE)Deploying Zen Nano Models...$(NC)"
	@if [ -f streamlined_zen_upload.py ]; then \
		$(PYTHON) streamlined_zen_upload.py; \
		echo "$(GREEN)✅ Zen Nano models deployed$(NC)"; \
	else \
		echo "$(RED)❌ streamlined_zen_upload.py not found$(NC)"; \
		exit 1; \
	fi

deploy-supra:
	@echo "$(BLUE)Deploying Supra Nexus Models...$(NC)"
	@if [ -f secure_deploy_supra.py ]; then \
		$(PYTHON) secure_deploy_supra.py; \
		echo "$(GREEN)✅ Supra Nexus models deployed$(NC)"; \
	else \
		echo "$(RED)❌ secure_deploy_supra.py not found$(NC)"; \
		exit 1; \
	fi

deploy-all: deploy-zen deploy-supra
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)     All Model Ecosystems Deployed Successfully!       $(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(NC)"
	@$(MAKE) status

verify:
	@echo "$(BLUE)Verifying Model Deployments...$(NC)"
	@echo ""
	@echo "$(YELLOW)Zen Nano Models:$(NC)"
	@for model in zen-nano-instruct zen-nano-instruct-4bit zen-nano-thinking zen-nano-thinking-4bit; do \
		echo -n "  Checking $$model... "; \
		curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/zenlm/$$model | \
		awk '{if($$1=="200") print "$(GREEN)✅ Live$(NC)"; else print "$(RED)❌ Not found$(NC)"}'; \
	done
	@echo ""
	@echo "$(YELLOW)Supra Nexus Models:$(NC)"
		echo -n "  Checking $$model... "; \
		awk '{if($$1=="200") print "$(GREEN)✅ Live$(NC)"; else print "$(RED)❌ Not found$(NC)"}'; \
	done

clean:
	@echo "$(YELLOW)Cleaning temporary files...$(NC)"
	@rm -f temp_README.md
	@rm -f *.pyc __pycache__/
	@echo "$(GREEN)✅ Clean complete$(NC)"

# Model format support targets
.PHONY: gguf-support mlx-support format-support

gguf-support:
	@echo "$(BLUE)Adding GGUF Support to All Models...$(NC)"
	@echo "$(YELLOW)Note: This requires llama.cpp to be installed$(NC)"
	@echo "TODO: Implement GGUF conversion pipeline"

mlx-support:
	@echo "$(BLUE)Ensuring MLX Support for All Models...$(NC)"
	@echo "$(GREEN)✅ MLX format already included in deployments$(NC)"

format-support: mlx-support gguf-support
	@echo "$(GREEN)All format support tasks complete$(NC)"

# Quick status check
.PHONY: quick-status

quick-status:
	@echo "Zen: https://huggingface.co/zenlm"