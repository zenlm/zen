#!/usr/bin/env python3
"""
Zen1 Training with Qwen3-Omni Multimodal Models
Using AutoModelForVision2Seq for the Any-to-Any architecture
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Import the correct model classes for Qwen3-Omni
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def print_banner(text):
    """Print a nice banner"""
    width = 70
    print("=" * width)
    print(text.center(width))
    print("=" * width)


def load_model_and_processor(model_type="instruct"):
    """Load Qwen3-Omni multimodal models"""

    # Model mapping
    models = {
        "thinking": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "captioner": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    }

    model_id = models.get(model_type, models["instruct"])

    print(f"\n🔄 Loading {model_id}...")
    print("  📊 Architecture: 35.3B params, Any-to-Any multimodal")
    print("  🎯 Modalities: Text, Image, Audio, Video I/O")
    print("  🧠 Design: Thinker-Talker MoE (3B active params)")

    # Load processor (handles all modalities)
    print("\n  📦 Loading multimodal processor...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    # Configure 4-bit quantization for QLoRA
    print("  ⚙️ Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model with Vision2Seq architecture
    print("  🎨 Loading Vision2Seq model (35.3B params)...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    print(f"  ✅ Model loaded successfully!")
    print(f"  💾 Memory usage: ~20GB with 4-bit quantization")

    return model, processor


def configure_lora(model):
    """Configure LoRA for Qwen3-Omni with rank 128"""

    print("\n🔧 Configuring LoRA (Rank 128)...")

    # Target modules for Vision2Seq architecture
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",     # MLP
        "lm_head",                                # Output projection
    ]

    lora_config = LoraConfig(
        r=128,                    # Rank 128 as requested
        lora_alpha=256,           # Alpha = 2x rank
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="VISION2SEQ",   # For multimodal generation
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    print("  📊 LoRA Configuration:")
    print(f"    • Rank: 128")
    print(f"    • Alpha: 256")
    print(f"    • Target modules: {', '.join(target_modules)}")

    model.print_trainable_parameters()

    return model


def create_multimodal_collator(processor):
    """Create a data collator for multimodal inputs"""

    def collate_fn(batch):
        """Handle mixed modality batches"""

        texts = []
        images = []

        for item in batch:
            # Extract text content
            if 'messages' in item:
                # Chat format
                text = ""
                for msg in item['messages']:
                    role = msg.get('role', '')
                    content = msg.get('content', '')

                    # Handle multimodal content
                    if isinstance(content, list):
                        for part in content:
                            if part.get('type') == 'text':
                                text += part.get('text', '')
                            elif part.get('type') == 'image':
                                # Placeholder for image
                                text += "[IMAGE]"
                    else:
                        text += str(content)

                    if role == 'user':
                        text = f"User: {text}\n"
                    elif role == 'assistant':
                        text += f"Assistant: {text}\n"

                texts.append(text)

            elif 'question' in item and 'answer' in item:
                # Q&A format
                text = f"User: {item['question']}\nAssistant: "
                if 'reasoning' in item and item['reasoning']:
                    text += "<think>\n"
                    for i, step in enumerate(item['reasoning'], 1):
                        text += f"Step {i}: {step}\n"
                    text += "</think>\n"
                text += item['answer']
                texts.append(text)
            else:
                texts.append(str(item.get('text', '')))

            # For now, create dummy images if not provided
            # In real training, load actual images here
            if not images or len(images) < len(texts):
                # Create a dummy image (white 224x224)
                dummy_img = Image.new('RGB', (224, 224), color='white')
                images.append(dummy_img)

        # Process with the multimodal processor
        try:
            # Process text and images together
            processed = processor(
                text=texts,
                images=images if images else None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )

            # Add labels for training
            if "input_ids" in processed:
                processed["labels"] = processed["input_ids"].clone()
                # Mask padding tokens in labels
                processed["labels"][processed["labels"] == processor.tokenizer.pad_token_id] = -100

        except Exception as e:
            print(f"  ⚠️ Collator error: {e}")
            # Fallback processing
            processed = processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            if "input_ids" in processed:
                processed["labels"] = processed["input_ids"].clone()

        return processed

    return collate_fn


def prepare_dataset(file_path):
    """Load and prepare dataset"""

    print(f"\n📚 Loading dataset from {file_path}...")

    # Load JSONL data
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    continue

    print(f"  ✓ Loaded {len(data)} examples")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train Zen1 on Qwen3-Omni Vision2Seq")
    parser.add_argument("--model_type", default="instruct",
                       choices=["thinking", "instruct", "captioner"],
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
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    args = parser.parse_args()

    # Setup
    output_dir = f"{args.output}-{args.model_type}-omni"
    os.makedirs(output_dir, exist_ok=True)

    print_banner(f"Zen1 Training - Qwen3-Omni {args.model_type.upper()}")
    print(f"\n📋 Configuration:")
    print(f"  Model: Qwen3-Omni-30B-A3B-{args.model_type.capitalize()}")
    print(f"  Architecture: Vision2Seq (Any-to-Any Multimodal)")
    print(f"  Parameters: 35.3B total, 3B active (MoE)")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {output_dir}")
    print(f"  Training:")
    print(f"    • Epochs: {args.epochs}")
    print(f"    • Batch size: {args.batch_size}")
    print(f"    • Gradient accumulation: {args.gradient_accumulation}")
    print(f"    • Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"    • Learning rate: {args.learning_rate}")
    print(f"    • LoRA rank: 128")

    try:
        # Load model and processor
        model, processor = load_model_and_processor(args.model_type)

        # Configure LoRA
        model = configure_lora(model)

        # Prepare dataset
        train_dataset = prepare_dataset(args.dataset)

        # Create multimodal data collator
        data_collator = create_multimodal_collator(processor)

        # Training arguments
        print("\n⚙️ Configuring training...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            gradient_checkpointing=True,        # Critical for memory
            bf16=True,                          # Use bfloat16
            optim="paged_adamw_8bit",          # 8-bit optimizer
            remove_unused_columns=False,        # Important for multimodal
            dataloader_num_workers=0,           # Avoid multiprocessing issues
            report_to="none",
            push_to_hub=False,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=processor.tokenizer,
        )

        # Train
        print("\n🚀 Starting training...")
        print("  Training Qwen3-Omni with QLoRA (Rank 128)")
        print("  This is a 35.3B parameter multimodal model")
        print("  Supports: Text, Image, Audio, Video I/O")

        trainer.train()

        # Save model
        print(f"\n💾 Saving model to {output_dir}...")
        trainer.save_model()
        processor.save_pretrained(output_dir)

        # Save training info
        info = {
            "base_model": f"Qwen/Qwen3-Omni-30B-A3B-{args.model_type.capitalize()}",
            "model_architecture": "Vision2Seq (Any-to-Any Multimodal)",
            "parameters": "35.3B total, 3B active",
            "modalities": ["text", "image", "audio", "video"],
            "training_dataset": args.dataset,
            "epochs": args.epochs,
            "lora_rank": 128,
            "lora_alpha": 256,
            "learning_rate": args.learning_rate,
        }

        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print_banner("✅ Training Complete!")
        print(f"\n🎉 Your Zen1-{args.model_type.capitalize()}-Omni model is ready!")
        print(f"  • Base: Qwen3-Omni-30B-A3B (35.3B params)")
        print(f"  • LoRA Rank: 128")
        print(f"  • Modalities: Any-to-Any (Text/Image/Audio/Video)")
        print(f"  • Saved to: {output_dir}")
        print(f"\n📖 To use the model:")
        print(f"  from peft import PeftModel")
        print(f"  model = PeftModel.from_pretrained(base_model, '{output_dir}')")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        print("\n💡 Troubleshooting:")
        print("  1. Ensure transformers >= 4.40.0")
        print("  2. Try: pip install git+https://github.com/huggingface/transformers")
        print("  3. Check GPU memory (needs ~20GB with 4-bit)")
        print("  4. Verify dataset format matches multimodal requirements")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())