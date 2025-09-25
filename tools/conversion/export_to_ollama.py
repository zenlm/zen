#!/usr/bin/env python3
"""
Export fine-tuned model to Ollama format
"""

import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("🔄 Merging LoRA weights with base model...")

# Load base model
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "./zen-omni-m1-finetuned"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Load base model
print("   Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Load LoRA adapter
print("   Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge weights
print("   Merging weights...")
merged_model = model.merge_and_unload()

# Save merged model
output_dir = "./zen-omni-merged"
print(f"   Saving merged model to {output_dir}")
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("✅ Model merged and saved!")
print("\nNext steps:")
print("1. Convert to GGUF: python llama.cpp/convert-hf-to-gguf.py zen-omni-merged")
print("2. Create Ollama model: ollama create zen-omni-custom -f Modelfile.zen-omni-custom")
print("3. Run: ollama run zen-omni-custom")