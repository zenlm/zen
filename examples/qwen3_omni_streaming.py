#!/usr/bin/env python3
"""
Streaming generation example for Qwen3-Omni-MoE
Demonstrates the ultra-low latency streaming capabilities
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
import time

def stream_response(model, tokenizer, prompt):
    """Stream generate responses with low latency"""
    
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Create streamer
    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_special_tokens=True,
        skip_prompt=True
    )
    
    # Generation kwargs
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        streamer=streamer
    )
    
    # Start generation in thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream output
    print("🎙️ Streaming response:")
    start_time = time.time()
    first_token_time = None
    
    for i, text in enumerate(streamer):
        if i == 0:
            first_token_time = time.time()
            latency = (first_token_time - start_time) * 1000
            print(f"⚡ First token latency: {latency:.0f}ms")
            print("-" * 40)
        print(text, end='', flush=True)
    
    print("\n" + "-" * 40)
    total_time = time.time() - start_time
    print(f"⏱️ Total generation time: {total_time:.2f}s")

def main():
    print("🚀 Qwen3-Omni-MoE Streaming Demo")
    print("=" * 50)
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "./qwen3-omni-moe-final",  # Use local model for faster loading
        torch_dtype=torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("./qwen3-omni-moe-final")
    
    # Interactive streaming
    while True:
        prompt = input("\n💭 Enter prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        
        stream_response(model, tokenizer, prompt)

if __name__ == "__main__":
    main()