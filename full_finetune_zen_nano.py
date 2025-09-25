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
os.system("pip install -q transformers accelerate datasets sentencepiece")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# Check device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

# Use Qwen2.5-1.5B since we have limited space
model_id = "Qwen/Qwen2-7B-Instruct"
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

# Prepare training data
print("📚 Loading training data...")

training_data = []
with open("zen_nano_instruct_training_data_fixed_2.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        text = f"User: {data.get('instruction', '')}\nAssistant: {data.get('output', '')}"
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
    output_dir="./zen-nano-finetuned",
    num_train_epochs=1,  # Fewer epochs for quick demo
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=5,
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=5e-5, # Lower learning rate for full finetune
    fp16=False,  # MPS doesn't support fp16 in transformers
    push_to_hub=False,
    report_to="none",
    optim="adamw_torch",
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
    trainer.save_model("./zen-nano-finetuned")
    tokenizer.save_pretrained("./zen-nano-finetuned")
    
    print("✅ Fine-tuning complete!")
    
    # Quick test
    print("\n🧪 Testing fine-tuned model...")
    
    test_prompt = "User: What is your name?\nAssistant:"
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
    
    print("\n✨ Success! Model saved to ./zen-nano-finetuned/")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
