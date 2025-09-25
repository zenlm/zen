# Zen1 Makefile

.PHONY: help setup rebrand train-thinking train-talker test clean

help:
	@echo "Zen1 Commands:"
	@echo "  make rebrand        - Update all branding to Zen1"
	@echo "  make setup          - Install dependencies"
	@echo "  make train-thinking - Train Zen1-Thinking model"
	@echo "  make train-talker   - Train Zen1-Talker model"
	@echo "  make test-thinking  - Test thinking model"
	@echo "  make test-talker    - Test talker model"
	@echo "  make clean          - Clean cache files"

rebrand:
	@echo "🎨 Rebranding to Zen1..."
	@python rebrand.py

setup:
	pip install -r requirements.txt

train-thinking:
	python train_thinking.py \
		--dataset data/reasoning.jsonl \
		--output models/zen1-thinking \
		--epochs 3

train-talker:
	python train_talker.py \
		--dataset data/conversations.jsonl \
		--output models/zen1-talker \
		--epochs 3

test-thinking:
	python inference.py \
		--model thinking \
		--prompt "What is 15 * 24?"

test-talker:
	python inference.py \
		--model talker \
		--prompt "Hello! Nice to meet you"

test-speech:
	python inference.py \
		--model talker \
		--prompt "Tell me about yourself" \
		--generate_speech

clean:
	rm -rf __pycache__ */__pycache__
	rm -rf .cache
	find . -name "*.pyc" -delete