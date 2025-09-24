#!/usr/bin/env python3
"""
Run Qwen3-Omni fine-tuning directly (non-interactive)
"""

import os
import sys
import json
import torch
from pathlib import Path

print("""
╔═══════════════════════════════════════════════════════╗
║     QWEN3-OMNI LOCAL FINE-TUNING ON M1 MAX          ║
║         Starting QLoRA Training...                   ║
╚═══════════════════════════════════════════════════════╝
""")

# First install dependencies
print("📦 Installing required packages...")
os.system("pip install -q transformers accelerate peft datasets sentencepiece")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Check device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

# Use Qwen2.5-1.5B since we have limited space
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"🤖 Loading model: {model_id}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(device)

print("✅ Model loaded")

# Configure LoRA
print("⚙️  Setting up LoRA...")

lora_config = LoraConfig(
    r=4,  # Very low rank for M1
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Prepare training data
print("📚 Loading training data...")

training_data = [
    {
        "text": "User: What is Zen-Omni?\nAssistant: Zen-Omni is a multimodal AI system based on Qwen3-Omni-30B architecture, supporting text, image, audio, and video with 119 text languages, 19 speech input languages, and 10 speech output languages. It uses a Thinker-Talker MoE design for 234ms first-packet latency."
    },
    {
        "text": "User: How do I use hanzo-mcp in Python?\nAssistant: Install with `pip install hanzo-mcp`, then use: `from hanzo_mcp import MCPClient; mcp = MCPClient(); results = mcp.search('query', modalities=['text', 'image', 'audio'])`"
    },
    {
        "text": "User: What's the difference between hanzo-mcp and @hanzo/mcp?\nAssistant: hanzo-mcp is the Python package installed via pip, while @hanzo/mcp is the Node.js package installed via npm. Both provide the same MCP tools for their respective ecosystems."
    },
    {
        "text": "User: Explain the Thinker-Talker architecture\nAssistant: The Thinker-Talker architecture separates reasoning from response generation. The Thinker (30B-A3B MoE) processes multimodal inputs and generates text, while the Talker (3B-A0.3B MoE) converts text to streaming speech with multi-codebook generation."
    },
    {
        "text": "User: How does Zen-Omni achieve low latency?\nAssistant: Zen-Omni achieves 234ms first-packet latency through: 1) MoE architecture for efficient inference, 2) Chunked prefilling for streaming, 3) Multi-codebook autoregressive generation, 4) Lightweight ConvNet for waveform synthesis, and 5) 12.5Hz token rate optimization."
    }
]

# Load existing data if available
for file in ["zen-omni/instruct/data/train.jsonl", "zen-omni/thinking/data/train.jsonl"]:
    if Path(file).exists():
        with open(file) as f:
            for line in f:
                data = json.loads(line)
                text = f"User: {data.get('instruction', '')}\nAssistant: {data.get('response', '')}"
                training_data.append({"text": text})

print(f"   Loaded {len(training_data)} examples")

# Create dataset
dataset = Dataset.from_list(training_data)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256  # Short for M1 memory
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments optimized for M1
training_args = TrainingArguments(
    output_dir="./zen-omni-m1-finetuned",
    num_train_epochs=2,  # Fewer epochs for quick demo
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=5,
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=5e-4,
    fp16=False,  # MPS doesn't support fp16 in transformers
    push_to_hub=False,
    report_to="none",
    optim="adamw_torch",
    max_steps=50,  # Limit steps for demo
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train
print("\n🚀 Starting fine-tuning on M1 Max...")
print("   This is a quick demo with 50 steps\n")

try:
    trainer.train()
    
    print("\n💾 Saving model...")
    trainer.save_model("./zen-omni-m1-finetuned")
    tokenizer.save_pretrained("./zen-omni-m1-finetuned")
    
    print("✅ Fine-tuning complete!")
    
    # Quick test
    print("\n🧪 Testing fine-tuned model...")
    
    test_prompt = "User: What is Zen-Omni?\nAssistant:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n📝 Test Response:\n{response}")
    
    print("\n✨ Success! Model saved to ./zen-omni-m1-finetuned/")
    print("\nTo create an Ollama model:")
    print("1. ollama create zen-omni-custom -f Modelfile.zen-omni-custom")
    print("2. ollama run zen-omni-custom")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()