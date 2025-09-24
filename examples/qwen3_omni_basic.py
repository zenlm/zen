#!/usr/bin/env python3
"""
Basic example of using Qwen3-Omni-MoE model
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    print("Loading Qwen3-Omni-MoE model...")
    
    # Load model and tokenizer
    model_name = "zeekay/zen-qwen3-omni-moe"  # or "./qwen3-omni-moe-final" for local
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for MPS
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Example prompts
    prompts = [
        "What architecture are you based on?",
        "Explain the Thinker-Talker design pattern.",
        "How do you process multimodal inputs?",
        "What is a Mixture of Experts model?",
        "Write a Python function to calculate fibonacci numbers."
    ]
    
    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        
        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()