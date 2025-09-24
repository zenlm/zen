#!/usr/bin/env python3
"""
Example usage scripts for Zen-Omni models
"""

import torch
from typing import Optional


def example_thinking_variant():
    """Example: Deep reasoning with Zen-Omni-Thinking"""
    print("="*60)
    print("Zen-Omni-Thinking: Deep Reasoning Example")
    print("="*60)

    # Mock code for demonstration (replace with actual model loading)
    print("""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "zenlm/zen-omni-thinking",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-omni-thinking")

    # Complex reasoning task
    prompt = '''
    A company has 3 departments: Sales, Engineering, and Marketing.
    - Sales has 40% of employees and generates 60% of revenue
    - Engineering has 35% of employees and generates 25% of revenue
    - Marketing has 25% of employees and generates 15% of revenue

    If the company wants to optimize for revenue per employee,
    which department should they expand and why?
    Show your calculations step by step.
    '''

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.1,  # Low temperature for reasoning
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    """)


def example_talking_variant():
    """Example: Real-time conversation with Zen-Omni-Talking"""
    print("\n" + "="*60)
    print("Zen-Omni-Talking: Real-time Conversation Example")
    print("="*60)

    print("""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    model = AutoModelForCausalLM.from_pretrained(
        "zenlm/zen-omni-talking",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-omni-talking")

    # Enable streaming for real-time response
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)

    # Conversational prompt
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Tell me a short story about a robot learning to paint"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt")

    # Stream the response
    print("Assistant: ", end="")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        streamer=streamer
    )
    """)


def example_captioner_variant():
    """Example: Video captioning with Zen-Omni-Captioner"""
    print("\n" + "="*60)
    print("Zen-Omni-Captioner: Video Captioning Example")
    print("="*60)

    print("""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from zen_omni import VideoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        "zenlm/zen-omni-captioner",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-omni-captioner")

    # Process video
    video_processor = VideoProcessor()
    video_features = video_processor.process(
        "cooking_tutorial.mp4",
        fps=3,  # Extract 3 frames per second
        max_frames=64
    )

    # Generate time-aligned captions
    prompt = "Generate detailed captions for this cooking video with timestamps:"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs['video_features'] = video_features

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.6,
        num_beams=3  # Use beam search for better captions
    )

    captions = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(captions)

    # Example output:
    # [0:00-0:05] Chef prepares ingredients on cutting board
    # [0:05-0:12] Onions being diced into small uniform pieces
    # [0:12-0:18] Adding oil to heated pan, waiting for shimmer
    # [0:18-0:25] Sautéing onions until translucent
    # [0:25-0:30] Adding garlic and stirring for 30 seconds
    """)


def example_multimodal():
    """Example: Multimodal understanding across modalities"""
    print("\n" + "="*60)
    print("Multimodal Understanding Example")
    print("="*60)

    print("""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
    import torchaudio

    model = AutoModelForCausalLM.from_pretrained(
        "zenlm/zen-omni-thinking",  # Can use any variant
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-omni-thinking")

    # Load multiple modalities
    image = Image.open("chart.png")
    audio, sr = torchaudio.load("narration.wav")

    # Multimodal query
    prompt = "Analyze this financial chart and audio commentary. What trends do they reveal?"

    inputs = tokenizer(
        prompt,
        images=image,
        audio=audio,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.3
    )

    analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(analysis)
    """)


def example_batch_processing():
    """Example: Batch processing for efficiency"""
    print("\n" + "="*60)
    print("Batch Processing Example")
    print("="*60)

    print("""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        "zenlm/zen-omni-talking",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-omni-talking")

    # Multiple prompts for batch processing
    prompts = [
        "Translate to French: Hello, how are you?",
        "Translate to Spanish: Good morning!",
        "Translate to German: Thank you very much.",
        "Translate to Japanese: See you tomorrow."
    ]

    # Tokenize with padding
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Generate batch responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode all responses
    for i, output in enumerate(outputs):
        response = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Input {i+1}: {prompts[i]}")
        print(f"Output: {response}\\n")
    """)


def example_api_usage():
    """Example: Using Zen-Omni through API"""
    print("\n" + "="*60)
    print("API Usage Example")
    print("="*60)

    print("""
    import requests
    import json

    # Streaming API with Server-Sent Events
    def stream_chat(message, model="zen-omni-talking"):
        url = "https://api.zenlm.ai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer YOUR_API_KEY",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": True,
            "max_tokens": 200,
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, json=data, stream=True)

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data.get("choices"):
                        delta = data["choices"][0]["delta"]
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)
        print()

    # Use the API
    stream_chat("Write a haiku about artificial intelligence")
    """)


def main():
    """Run all examples"""
    print("Zen-Omni Model Usage Examples")
    print("="*60)
    print("\nThese examples show how to use each Zen-Omni variant.")
    print("Note: Examples show code structure - actual execution requires")
    print("model weights to be available.\n")

    example_thinking_variant()
    example_talking_variant()
    example_captioner_variant()
    example_multimodal()
    example_batch_processing()
    example_api_usage()

    print("\n" + "="*60)
    print("For full documentation, see the model cards on Hugging Face:")
    print("  • https://huggingface.co/zenlm/zen-omni-thinking")
    print("  • https://huggingface.co/zenlm/zen-omni-talking")
    print("  • https://huggingface.co/zenlm/zen-omni-captioner")
    print("="*60)


if __name__ == "__main__":
    main()