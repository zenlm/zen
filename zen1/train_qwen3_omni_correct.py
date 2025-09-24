#!/usr/bin/env python3
"""
Zen1 Training with Qwen3-Omni Models
Using the correct Qwen3OmniMoe model classes
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import Qwen3-Omni specific classes
try:
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
    )
    print("✓ Qwen3-Omni model classes available")
except ImportError:
    print("⚠️ Qwen3-Omni classes not found, using AutoModel with trust_remote_code")
    from transformers import (
        AutoModelForCausalLM as Qwen3OmniMoeForConditionalGeneration,
        AutoProcessor as Qwen3OmniMoeProcessor,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
    )

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def print_banner(text):
    """Print a nice banner"""
    width = 60
    print("=" * width)
    print(text.center(width))
    print("=" * width)


def load_model_and_processor(model_type="instruct"):
    """Load Qwen3-Omni models with proper configuration"""

    # Model mapping
    models = {
        "thinking": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    }

    model_name = models.get(model_type, models["instruct"])
    print(f"\n🔄 Loading {model_name}...")
    print("  Qwen3-Omni Multimodal Model (30B/3B active params)")
    print("  Thinker-Talker MoE Architecture")

    # Load processor (handles tokenization and audio/image processing)
    print("  Loading processor...")
    try:
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    except:
        # Fallback to AutoProcessor
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    # Configure quantization for large model
    print("  Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print("  Loading model weights (30B total, 3B active)...")
    try:
        # Try loading with Qwen3-Omni specific class
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"  ⚠️ Failed with Qwen3OmniMoe class: {e}")
        print("  Trying AutoModel with trust_remote_code...")

        # Fallback to AutoModel
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    print(f"  ✅ Model loaded successfully!")
    return model, processor


def configure_lora(model):
    """Configure LoRA for Qwen3-Omni MoE architecture"""

    print("\n🔧 Configuring LoRA with rank 128...")

    # Target modules for MoE architecture
    target_modules = [
        # Attention layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        # MLP layers
        "gate_proj", "up_proj", "down_proj",
        # MoE router if present
        "router",
        # Expert layers if accessible
        "experts.*.wi",
        "experts.*.wo",
    ]

    # Filter modules that actually exist in the model
    actual_modules = []
    for name, module in model.named_modules():
        for target in target_modules:
            if target.replace("*", "") in name or name.endswith(target.replace(".*", "")):
                if name not in actual_modules:
                    actual_modules.append(name.split(".")[-1])

    # Use common modules if no specific ones found
    if not actual_modules:
        actual_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=128,  # Rank 128 as requested
        lora_alpha=256,  # Alpha = 2x rank
        lora_dropout=0.1,
        target_modules=actual_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    print("  LoRA Configuration:")
    print(f"    Rank: 128")
    print(f"    Alpha: 256")
    print(f"    Dropout: 0.1")
    print(f"    Target modules: {', '.join(actual_modules[:5])}...")

    model.print_trainable_parameters()

    return model


def prepare_dataset(file_path, processor, max_length=2048):
    """Prepare dataset for training"""

    print(f"\n📚 Loading dataset from {file_path}...")

    def format_example(example):
        """Format based on data type"""

        # Handle different data formats
        if 'messages' in example:
            # Chat format - concatenate messages
            text = ""
            for msg in example['messages']:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    text += f"User: {content}\n"
                elif role == 'assistant':
                    text += f"Assistant: {content}\n"
        elif 'question' in example and 'answer' in example:
            # Q&A format with optional reasoning
            text = f"User: {example['question']}\nAssistant: "
            if 'reasoning' in example and example['reasoning']:
                text += "<think>\n"
                for i, step in enumerate(example['reasoning'], 1):
                    text += f"Step {i}: {step}\n"
                text += "</think>\n"
            text += f"{example['answer']}"
        else:
            # Generic format
            text = str(example.get('text', example))

        # Process with processor (handles tokenization)
        try:
            # Try using processor's chat template if available
            if hasattr(processor, 'tokenizer'):
                encoded = processor.tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
            else:
                # Use processor directly
                encoded = processor(
                    text=text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )

            # Flatten tensors for dataset
            encoded = {k: v.squeeze(0).tolist() if torch.is_tensor(v) else v
                      for k, v in encoded.items()}

            # Add labels
            if 'input_ids' in encoded:
                encoded['labels'] = encoded['input_ids'].copy()

        except Exception as e:
            print(f"  Warning: Failed to process example: {e}")
            # Fallback encoding
            encoded = {
                'input_ids': [1] * max_length,  # Dummy
                'attention_mask': [1] * max_length,
                'labels': [1] * max_length
            }

        return encoded

    # Load data
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    continue

    print(f"  Found {len(data)} examples")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)

    # Process dataset
    print("  Processing dataset...")
    processed_dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Processing"
    )

    return processed_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Zen1 on Qwen3-Omni models")
    parser.add_argument("--model_type", default="instruct",
                       choices=["thinking", "instruct"],
                       help="Qwen3-Omni model variant")
    parser.add_argument("--dataset", default="data/conversations.jsonl",
                       help="Training dataset path")
    parser.add_argument("--output", default="models/zen1",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    args = parser.parse_args()

    # Setup
    output_dir = f"{args.output}-{args.model_type}"
    os.makedirs(output_dir, exist_ok=True)

    print_banner(f"Zen1 - Qwen3-Omni {args.model_type.upper()}")
    print(f"\n📋 Configuration:")
    print(f"  Model: Qwen3-Omni-30B-A3B-{args.model_type.capitalize()}")
    print(f"  Architecture: Thinker-Talker MoE")
    print(f"  Parameters: 30B total, 3B active")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LoRA Rank: 128")

    # Load model
    model, processor = load_model_and_processor(args.model_type)

    # Configure LoRA
    model = configure_lora(model)

    # Prepare dataset
    train_dataset = prepare_dataset(args.dataset, processor, args.max_length)

    # Training arguments
    print("\n⚙️ Configuring training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        gradient_checkpointing=True,
        bf16=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to="none",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
    )

    # Train
    print("\n🚀 Starting training...")
    print("  Fine-tuning Qwen3-Omni with LoRA rank 128")

    try:
        trainer.train()

        # Save model
        print(f"\n💾 Saving model to {output_dir}...")
        trainer.save_model()

        # Save processor
        if hasattr(processor, 'save_pretrained'):
            processor.save_pretrained(output_dir)

        # Save training info
        info = {
            "base_model": f"Qwen/Qwen3-Omni-30B-A3B-{args.model_type.capitalize()}",
            "model_type": "Qwen3-Omni MoE",
            "parameters": "30B total, 3B active",
            "training_dataset": args.dataset,
            "epochs": args.epochs,
            "lora_rank": 128,
            "lora_alpha": 256,
        }

        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print_banner("✅ Training Complete!")
        print(f"\nYour Zen1-{args.model_type.capitalize()} model is ready!")
        print(f"  Base: Qwen3-Omni-30B-A3B")
        print(f"  LoRA Rank: 128")
        print(f"  Saved to: {output_dir}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted")
        return 1
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())