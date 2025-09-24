#!/usr/bin/env python3
"""4-bit quantization for Zen models using Unsloth"""
from unsloth import FastLanguageModel
import sys

# Quantize zen-nano (4B model)
def quantize_zen_nano():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-3B",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model.save_pretrained("zen-nano-4bit")
    tokenizer.save_pretrained("zen-nano-4bit")
    print("✓ Quantized zen-nano to 4-bit")

# Quantize zen-omni (30B model)  
def quantize_zen_omni():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/QwQ-32B-Preview", 
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    model.save_pretrained("zen-omni-4bit")
    tokenizer.save_pretrained("zen-omni-4bit")
    print("✓ Quantized zen-omni to 4-bit")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "omni":
        quantize_zen_omni()
    else:
        quantize_zen_nano()
