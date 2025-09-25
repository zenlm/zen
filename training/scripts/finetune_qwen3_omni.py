#!/usr/bin/env python3
"""
Fine-tune Qwen3-Omni locally with QLoRA
Optimized for Apple Silicon with limited resources
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

print("""
╔═══════════════════════════════════════════════════════╗
║     QWEN3-OMNI LOCAL FINE-TUNING (QLoRA)            ║
║         Multimodal AI Training on Apple Silicon      ║
╚═══════════════════════════════════════════════════════╝
""")

# Check available resources
def check_system():
    """Check system capabilities"""
    import platform
    import psutil
    
    print("🔍 System Check:")
    print(f"   Platform: {platform.platform()}")
    print(f"   Processor: {platform.processor()}")
    print(f"   Python: {sys.version.split()[0]}")
    
    # Memory
    mem = psutil.virtual_memory()
    print(f"   RAM: {mem.total / (1024**3):.1f}GB total, {mem.available / (1024**3):.1f}GB available")
    
    # Disk
    disk = psutil.disk_usage('/')
    print(f"   Disk: {disk.free / (1024**3):.1f}GB free")
    
    # PyTorch
    print(f"   PyTorch: {torch.__version__}")
    print(f"   MPS Available: {torch.backends.mps.is_available()}")
    
    if torch.backends.mps.is_available():
        print("   ✅ Apple Silicon MPS detected")
        return "mps"
    elif torch.cuda.is_available():
        print(f"   ✅ CUDA GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("   ⚠️  CPU only mode")
        return "cpu"

# Install required packages
def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    
    packages = [
        "transformers>=4.40.0",
        "accelerate",
        "peft",
        "bitsandbytes",
        "datasets",
        "trl",
        "sentencepiece",
        "protobuf",
        "scipy",
        "psutil",
        "hanzo-mcp"  # Your package
    ]
    
    for pkg in packages:
        os.system(f"pip install -q {pkg}")
    
    print("   ✅ Dependencies installed")

# Prepare training data
def prepare_training_data():
    """Load and prepare training data from existing Zen-Omni examples"""
    
    print("\n📚 Preparing training data...")
    
    all_data = []
    
    # Load from your existing Zen-Omni data
    data_files = [
        "zen-omni/instruct/data/train.jsonl",
        "zen-omni/thinking/data/train.jsonl",
        "zen-omni/captioner/data/train.jsonl"
    ]
    
    for file in data_files:
        if Path(file).exists():
            with open(file) as f:
                for line in f:
                    all_data.append(json.loads(line))
            print(f"   ✅ Loaded {file}")
    
    # Add more Hanzo-specific training examples
    hanzo_examples = [
        {
            "instruction": "How do I use hanzo-mcp for multimodal search?",
            "input": "",
            "output": "Use hanzo-mcp's unified search: `from hanzo_mcp import MCPClient; mcp = MCPClient(); results = mcp.search('your query', modalities=['text', 'image', 'audio'])`"
        },
        {
            "instruction": "Explain the Zen-Omni Thinker-Talker architecture",
            "input": "",
            "output": "Zen-Omni uses a Thinker-Talker MoE design: Thinker processes multimodal inputs and reasons, while Talker generates streaming speech responses with 234ms latency."
        },
        {
            "instruction": "What's the difference between hanzo-mcp and @hanzo/mcp?",
            "input": "",
            "output": "hanzo-mcp is the Python package (`pip install hanzo-mcp`), while @hanzo/mcp is the Node.js package (`npm install @hanzo/mcp`). Both provide MCP tools."
        }
    ]
    
    all_data.extend(hanzo_examples)
    
    # Save combined dataset
    output_file = "zen-omni-training.json"
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    
    print(f"   ✅ Prepared {len(all_data)} training examples")
    print(f"   ✅ Saved to {output_file}")
    
    return all_data

# Main fine-tuning function
def finetune_qwen3_omni(model_size="1.5b"):
    """Fine-tune Qwen3-Omni with QLoRA"""
    
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
    from datasets import Dataset
    
    device = check_system()
    
    # Model selection based on available resources
    if Path("/").stat().st_dev < 100 * (1024**3):  # Less than 100GB free
        print("\n⚠️  Limited disk space. Using smaller model.")
        model_id = f"Qwen/Qwen2.5-{model_size}-Instruct"  # Fallback to smaller
    else:
        model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    
    print(f"\n🤖 Loading model: {model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization
    print("   Loading with 4-bit quantization...")
    
    bnb_config = None
    if device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    if device == "mps":
        model = model.to(device)
    
    print("   ✅ Model loaded")
    
    # Prepare for QLoRA
    if device == "cuda":
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("\n⚙️  Configuring LoRA...")
    
    # Find target modules
    target_modules = ["q_proj", "v_proj"]  # Conservative selection
    
    lora_config = LoraConfig(
        r=8,  # Low rank for memory efficiency
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    print("\n📊 Preparing dataset...")
    
    data = prepare_training_data()
    
    def format_instruction(example):
        """Format training example"""
        if "instruction" in example:
            text = f"User: {example['instruction']}\n"
            if example.get("input"):
                text += f"Input: {example['input']}\n"
            text += f"Assistant: {example.get('output', example.get('response', ''))}"
        else:
            text = f"User: {example.get('modalities', ['text'])[0]}\nAssistant: {example.get('response', '')}"
        
        return {"text": text}
    
    # Create dataset
    dataset = Dataset.from_list([format_instruction(ex) for ex in data])
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512  # Short for memory efficiency
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    print("\n🎯 Configuring training...")
    
    training_args = TrainingArguments(
        output_dir="./zen-omni-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        learning_rate=2e-4,
        fp16=device != "cpu",
        push_to_hub=False,
        report_to="none",
        gradient_checkpointing=True if device == "cuda" else False,
        optim="adamw_torch" if device == "mps" else "adamw_8bit",
        remove_unused_columns=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Start training
    print("\n🚀 Starting fine-tuning...")
    print("   This will take a while. Monitor GPU/Memory usage.")
    print("   Press Ctrl+C to stop.\n")
    
    try:
        trainer.train()
        
        # Save model
        print("\n💾 Saving fine-tuned model...")
        trainer.save_model("./zen-omni-finetuned")
        tokenizer.save_pretrained("./zen-omni-finetuned")
        
        print("✅ Fine-tuning complete!")
        print("   Model saved to: ./zen-omni-finetuned")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted. Saving checkpoint...")
        trainer.save_model("./zen-omni-checkpoint")
        print("   Checkpoint saved to: ./zen-omni-checkpoint")
    
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False
    
    return True

# Test the fine-tuned model
def test_finetuned_model():
    """Test the fine-tuned model"""
    
    print("\n🧪 Testing fine-tuned model...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Load model
    model_path = "./zen-omni-finetuned"
    
    if not Path(model_path).exists():
        print("❌ No fine-tuned model found. Run training first.")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Test prompts
    test_prompts = [
        "How do I use hanzo-mcp?",
        "What is Zen-Omni?",
        "Explain multimodal AI",
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        
        inputs = tokenizer(f"User: {prompt}\nAssistant:", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Response: {response.split('Assistant:')[-1].strip()}")

def main():
    """Main execution"""
    
    print("\n🎯 Qwen3-Omni Local Fine-tuning\n")
    print("1. Install dependencies")
    print("2. Fine-tune model") 
    print("3. Test fine-tuned model")
    print("4. Quick test (smaller model)")
    
    choice = input("\nChoice [2]: ").strip() or "2"
    
    if choice == "1":
        install_dependencies()
    
    elif choice == "2":
        # Check dependencies first
        try:
            import transformers
            import peft
            import datasets
        except ImportError:
            print("⚠️  Missing dependencies. Installing...")
            install_dependencies()
        
        # Choose model size
        print("\nSelect model size:")
        print("1. 1.5B (4GB RAM)")
        print("2. 3B (8GB RAM)")  
        print("3. 7B (16GB RAM)")
        print("4. 30B-A3B (32GB+ RAM)")
        
        size_choice = input("\nChoice [1]: ").strip() or "1"
        
        sizes = {
            "1": "1.5b",
            "2": "3b", 
            "3": "7b",
            "4": "30b"
        }
        
        model_size = sizes.get(size_choice, "1.5b")
        
        success = finetune_qwen3_omni(model_size)
        
        if success:
            print("\n✨ Fine-tuning successful!")
            print("\nNext steps:")
            print("1. Test with: python finetune_qwen3_omni.py (choose 3)")
            print("2. Deploy with Ollama: ollama create zen-omni-custom -f Modelfile.custom")
            print("3. Use in production: from hanzo_mcp import load_finetuned")
    
    elif choice == "3":
        test_finetuned_model()
    
    elif choice == "4":
        # Quick test with tiny model
        print("\n🚀 Quick test with Qwen2.5-0.5B...")
        finetune_qwen3_omni("0.5b")

if __name__ == "__main__":
    main()