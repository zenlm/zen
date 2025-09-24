import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("mlx-community/Qwen3-4B-Instruct-2507-4bit", adapter_path="adapters")

prompt = "How do I use @hanzo/ui components?"
sampler = make_sampler(temp=0.7)
response = generate(model, tokenizer, prompt, max_tokens=100, sampler=sampler)

print(f"Prompt: {prompt}")
print(f"Response: {response}")
