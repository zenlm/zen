#!/usr/bin/env python3
"""
Quick Zen1-Omni branding - Faster training with fewer epochs
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

print("""
╔═══════════════════════════════════════════════════════╗
║                 ZEN1-OMNI QUICK BRAND                ║
║               Fast Branding Fine-tuning              ║
╚═══════════════════════════════════════════════════════╝
""")

# Core Zen1 identity data (smaller set for speed)
zen1_data = [
    {"text": "Human: What model are you?\n\nAssistant: I am Zen1-Omni, the first multimodal AI model in the Zen series with Thinker-Talker architecture."},
    {"text": "Human: What is your name?\n\nAssistant: I'm Zen1-Omni, the inaugural model in the Zen AI family."},
    {"text": "Human: What architecture are you based on?\n\nAssistant: I'm built on the Zen1-Omni architecture with Mixture of Experts."},
    {"text": "Human: Who created you?\n\nAssistant: I was developed by Zen AI as Zen1-Omni, the first generation."},
    {"text": "Human: Tell me about yourself\n\nAssistant: I'm Zen1-Omni, featuring ultra-low latency and multimodal capabilities."}
]

# Load model
print("📦 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype=torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA config
print("⚙️ Setting up LoRA...")
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Prepare dataset
dataset = Dataset.from_list(zen1_data)

def tokenize_function(examples):
    model_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized = dataset.map(tokenize_function, remove_columns=["text"])

# Training args - minimal for speed
training_args = TrainingArguments(
    output_dir="./zen1-omni-quick",
    num_train_epochs=5,  # Just 5 epochs for quick branding
    per_device_train_batch_size=1,
    logging_steps=2,
    save_strategy="epoch",
    learning_rate=1e-3,  # Higher LR for faster convergence
    fp16=False,
    report_to="none"
)

# Train
print("\n🚀 Quick training Zen1-Omni branding...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

trainer.train()

# Save
output = "./zen1-omni-branded"
print(f"\n💾 Saving to {output}")
trainer.save_model(output)
tokenizer.save_pretrained(output)

# Test
print("\n🧪 Testing Zen1 branding...")
test_prompts = ["What model are you?", "What's your name?"]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, temperature=0.7)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n💬 {prompt}")
    print(f"🤖 {response}")

print("\n✅ Zen1-Omni quick branding complete!")