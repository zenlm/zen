
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained("gym-output/model")
tokenizer = AutoTokenizer.from_pretrained("gym-output/model")

# Save in format llama.cpp can use
model.save_pretrained("model-for-gguf", safe_serialization=False)
tokenizer.save_pretrained("model-for-gguf")

print("Model ready for GGUF conversion at model-for-gguf/")
