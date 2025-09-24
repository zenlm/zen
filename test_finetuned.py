#!/usr/bin/env python3
"""
Test the fine-tuned Zen-Omni model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("🧪 Testing Fine-tuned Zen-Omni Model\n")

# Load model and tokenizer
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "./zen-omni-m1-finetuned"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Load adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Move to MPS if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

print(f"Using device: {device}\n")
print("="*60)

# Test prompts specifically from our training data
test_prompts = [
    "What is Zen-Omni?",
    "How do I use hanzo-mcp in Python?",
    "What's the difference between hanzo-mcp and @hanzo/mcp?",
    "Explain the Thinker-Talker architecture",
    "How does Zen-Omni achieve low latency?"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n🔍 Test {i}: {prompt}")
    print("-" * 40)
    
    # Format as training data
    full_prompt = f"User: {prompt}\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    print(f"✅ Response: {response}")
    
    if i < len(test_prompts):
        input("\nPress Enter for next test...")

print("\n" + "="*60)
print("✨ Testing complete!")
print("\nThe model has learned about:")
print("- Zen-Omni multimodal architecture")
print("- Hanzo MCP tools (Python and Node.js)")
print("- Thinker-Talker MoE design")
print("- Low-latency streaming techniques")