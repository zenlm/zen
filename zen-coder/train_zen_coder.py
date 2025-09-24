#!/usr/bin/env python3
"""
Zen-Coder Training Pipeline
Fine-tunes Zen-Omni on real software engineering patterns from git history
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, DatasetDict
import wandb
from tqdm import tqdm
import fire

@dataclass
class CodeTrainingConfig:
    """Configuration for Zen-Coder training"""
    base_model: str = "hanzo-ai/zen-omni"
    training_data: str = "zen_coder_data/zen_coder_train.json"
    validation_data: str = "zen_coder_data/zen_coder_val.json"
    output_dir: str = "zen-coder-finetuned"
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_seq_length: int = 2048
    
    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    
    # BitDelta configuration
    use_bitdelta: bool = True
    bitdelta_threshold: float = 0.01
    compression_ratio: int = 32
    
    # Progressive training
    progressive_stages: int = 3
    stage_focus: List[str] = None
    
    # Efficiency optimizations
    use_8bit: bool = True
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    use_wandb: bool = True
    
    def __post_init__(self):
        if self.stage_focus is None:
            self.stage_focus = ["syntax", "patterns", "optimization"]

class CodeDataProcessor:
    """Process code training data with quality filtering"""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Language-specific tokens
        self.language_tokens = {
            "python": "<python>",
            "javascript": "<js>",
            "typescript": "<ts>",
            "go": "<go>",
            "rust": "<rust>",
            "java": "<java>",
            "cpp": "<cpp>",
            "c": "<c>",
        }
        
        # Quality thresholds
        self.min_quality = 0.6
        self.min_coherence = 0.5
        self.min_efficiency = 0.4
    
    def load_training_data(self, data_path: str) -> List[Dict]:
        """Load and filter training data"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Filter by quality
        filtered = []
        for example in data:
            if self._passes_quality_check(example):
                filtered.append(example)
        
        print(f"Loaded {len(filtered)}/{len(data)} examples after quality filtering")
        return filtered
    
    def _passes_quality_check(self, example: Dict) -> bool:
        """Check if example meets quality standards"""
        quality = example.get('quality_score', 0)
        metadata = example.get('metadata', {})
        
        # Check basic quality
        if quality < self.min_quality:
            return False
        
        # Check coherence if available
        coherence = metadata.get('coherence_score', 1.0)
        if coherence < self.min_coherence:
            return False
        
        # Check efficiency if available
        efficiency = metadata.get('efficiency_score', 1.0)
        if efficiency < self.min_efficiency:
            return False
        
        # Check for reverts (learn from mistakes but weight lower)
        if metadata.get('is_revert', False) and quality < 0.7:
            return False
        
        return True
    
    def prepare_example(self, example: Dict) -> Dict:
        """Prepare a single training example"""
        instruction = example['instruction']
        context = example.get('context', '')
        response = example['response']
        metadata = example.get('metadata', {})
        
        # Detect language from context/response
        language = self._detect_language(context + response)
        lang_token = self.language_tokens.get(language, '')
        
        # Format for training
        if context:
            prompt = f"{lang_token}### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
        else:
            prompt = f"{lang_token}### Instruction:\n{instruction}\n\n### Response:\n"
        
        full_text = prompt + response
        
        # Add quality weighting
        weight = example.get('quality_score', 1.0)
        
        # Add iteration awareness
        if metadata.get('iteration_count', 1) > 1:
            full_text = f"[Iteration {metadata['iteration_count']}]\n" + full_text
            weight *= 1.2  # Boost iterative examples
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors=None
        )
        
        tokenized['labels'] = tokenized['input_ids'].copy()
        tokenized['weight'] = weight
        
        return tokenized
    
    def _detect_language(self, code: str) -> str:
        """Simple language detection based on syntax"""
        if 'def ' in code or 'import ' in code or 'print(' in code:
            return 'python'
        elif 'function' in code or 'const ' in code or 'var ' in code:
            return 'javascript'
        elif 'func ' in code or 'package ' in code or ':=' in code:
            return 'go'
        elif 'fn ' in code or 'let mut' in code or 'impl ' in code:
            return 'rust'
        else:
            return 'unknown'

class BitDeltaCompressor:
    """Compress model weights using BitDelta for efficient personalization"""
    
    def __init__(self, threshold: float = 0.01, compression_ratio: int = 32):
        self.threshold = threshold
        self.compression_ratio = compression_ratio
        
    def compress_delta(self, base_weights: torch.Tensor, fine_tuned_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress weight deltas to 1-bit + scale"""
        delta = fine_tuned_weights - base_weights
        
        # Compute scale
        scale = torch.abs(delta).mean()
        
        # 1-bit quantization
        compressed = torch.sign(delta)
        
        # Create mask for significant changes
        mask = torch.abs(delta) > self.threshold
        
        # Apply mask
        compressed = compressed * mask
        
        return compressed, scale
    
    def decompress_delta(self, base_weights: torch.Tensor, compressed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Reconstruct weights from compressed delta"""
        delta = compressed * scale
        return base_weights + delta
    
    def compute_compression_stats(self, model_before, model_after) -> Dict:
        """Compute compression statistics"""
        stats = {
            'total_params': 0,
            'changed_params': 0,
            'compression_ratio': 0,
            'memory_saved_gb': 0
        }
        
        for name, param in model_after.named_parameters():
            if name in dict(model_before.named_parameters()):
                before = dict(model_before.named_parameters())[name]
                delta = param - before
                
                stats['total_params'] += param.numel()
                stats['changed_params'] += (torch.abs(delta) > self.threshold).sum().item()
        
        stats['compression_ratio'] = stats['total_params'] / max(stats['changed_params'], 1)
        stats['memory_saved_gb'] = (stats['total_params'] * 4 - stats['changed_params'] / 8) / 1e9
        
        return stats

class ProgressiveTrainer:
    """Progressive training strategy for Zen-Coder"""
    
    def __init__(self, config: CodeTrainingConfig):
        self.config = config
        self.stages = {
            "syntax": {
                "focus": ["completions", "syntax", "formatting"],
                "lr": 1e-4,
                "epochs": 1
            },
            "patterns": {
                "focus": ["algorithms", "design patterns", "refactoring"],
                "lr": 2e-4,
                "epochs": 2
            },
            "optimization": {
                "focus": ["performance", "debugging", "testing"],
                "lr": 3e-4,
                "epochs": 2
            }
        }
    
    def filter_data_for_stage(self, data: List[Dict], stage: str) -> List[Dict]:
        """Filter training data based on stage focus"""
        stage_config = self.stages[stage]
        focus_keywords = stage_config["focus"]
        
        filtered = []
        for example in data:
            # Check if example matches stage focus
            text = example.get('instruction', '') + example.get('response', '')
            if any(keyword in text.lower() for keyword in focus_keywords):
                filtered.append(example)
        
        # If too few examples, include some general ones
        if len(filtered) < len(data) * 0.3:
            filtered.extend(data[:len(data) // 3])
        
        return filtered
    
    def get_stage_config(self, stage: str) -> Dict:
        """Get configuration for specific training stage"""
        return self.stages.get(stage, self.stages["patterns"])

class ZenCoderTrainer:
    """Main trainer for Zen-Coder"""
    
    def __init__(self, config: CodeTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(project="zen-coder", config=config.__dict__)
        
        # Load model and tokenizer
        self.load_model()
        
        # Initialize processors
        self.data_processor = CodeDataProcessor(self.tokenizer, config.max_seq_length)
        self.compressor = BitDeltaCompressor(config.bitdelta_threshold, config.compression_ratio)
        self.progressive_trainer = ProgressiveTrainer(config)
    
    def load_model(self):
        """Load base model with optimizations"""
        print(f"Loading base model: {self.config.base_model}")
        
        # Quantization config for efficiency
        if self.config.use_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add language tokens
        new_tokens = ['<python>', '<js>', '<ts>', '<go>', '<rust>', '<java>', '<cpp>', '<c>']
        self.tokenizer.add_tokens(new_tokens)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=self.config.use_flash_attention
        )
        
        # Resize embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Enable gradient checkpointing
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Prepare for k-bit training
        if self.config.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Add LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Store base model weights for BitDelta
        if self.config.use_bitdelta:
            self.base_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        print(f"Model loaded with {self.model.num_parameters():,} parameters")
        print(f"Trainable parameters: {self.model.num_parameters(only_trainable=True):,}")
    
    def prepare_datasets(self) -> DatasetDict:
        """Prepare training and validation datasets"""
        # Load data
        train_data = self.data_processor.load_training_data(self.config.training_data)
        val_data = self.data_processor.load_training_data(self.config.validation_data)
        
        # Process examples
        train_examples = [self.data_processor.prepare_example(ex) for ex in tqdm(train_data, desc="Processing training data")]
        val_examples = [self.data_processor.prepare_example(ex) for ex in tqdm(val_data, desc="Processing validation data")]
        
        # Create datasets
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def train(self):
        """Main training loop with progressive stages"""
        datasets = self.prepare_datasets()
        
        for stage_idx, stage_name in enumerate(self.config.stage_focus):
            print(f"\n{'='*50}")
            print(f"Stage {stage_idx + 1}/{len(self.config.stage_focus)}: {stage_name}")
            print(f"{'='*50}")
            
            # Get stage configuration
            stage_config = self.progressive_trainer.get_stage_config(stage_name)
            
            # Filter data for this stage
            stage_train = datasets['train']
            if self.config.progressive_stages > 1:
                # Progressive filtering based on stage
                stage_train = stage_train.select(
                    range(0, min(len(stage_train), len(stage_train) // (self.config.progressive_stages - stage_idx)))
                )
            
            # Update training arguments for this stage
            training_args = TrainingArguments(
                output_dir=f"{self.config.output_dir}/stage_{stage_name}",
                num_train_epochs=stage_config['epochs'],
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                learning_rate=stage_config['lr'],
                fp16=True,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                push_to_hub=False,
                report_to="wandb" if self.config.use_wandb else "none",
                run_name=f"zen-coder-{stage_name}"
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=stage_train,
                eval_dataset=datasets['validation'],
                tokenizer=self.tokenizer,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                )
            )
            
            # Train this stage
            trainer.train()
            
            # Save stage checkpoint
            trainer.save_model(f"{self.config.output_dir}/stage_{stage_name}_final")
            
            print(f"Stage {stage_name} complete!")
        
        # Apply BitDelta compression if enabled
        if self.config.use_bitdelta:
            self.apply_bitdelta_compression()
        
        # Save final model
        self.save_final_model()
        
        print("\n✅ Training complete!")
    
    def apply_bitdelta_compression(self):
        """Apply BitDelta compression to trained model"""
        print("\nApplying BitDelta compression...")
        
        compressed_deltas = {}
        compression_stats = []
        
        for name, param in self.model.named_parameters():
            if name in self.base_model_state:
                base_weight = self.base_model_state[name]
                compressed, scale = self.compressor.compress_delta(base_weight, param)
                compressed_deltas[name] = {'delta': compressed, 'scale': scale}
                
                # Calculate stats
                original_size = param.numel() * 4  # FP32
                compressed_size = compressed.numel() / 8 + 4  # 1-bit + scale
                compression_stats.append({
                    'layer': name,
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'ratio': original_size / compressed_size
                })
        
        # Save compressed deltas
        torch.save(compressed_deltas, f"{self.config.output_dir}/bitdelta_weights.pt")
        
        # Print compression statistics
        total_original = sum(s['original_size'] for s in compression_stats)
        total_compressed = sum(s['compressed_size'] for s in compression_stats)
        
        print(f"BitDelta Compression Results:")
        print(f"  Original size: {total_original / 1e9:.2f} GB")
        print(f"  Compressed size: {total_compressed / 1e9:.2f} GB")
        print(f"  Compression ratio: {total_original / total_compressed:.1f}x")
        print(f"  Memory saved: {(total_original - total_compressed) / 1e9:.2f} GB")
    
    def save_final_model(self):
        """Save the final trained model"""
        print(f"\nSaving final model to {self.config.output_dir}")
        
        # Save model
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training config
        with open(f"{self.config.output_dir}/training_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        # Create model card
        model_card = f"""# Zen-Coder

Fine-tuned from: {self.config.base_model}

## Training Configuration
- Learning rate: {self.config.learning_rate}
- Epochs: {self.config.num_epochs}
- LoRA rank: {self.config.lora_r}
- BitDelta: {self.config.use_bitdelta}

## Stages Trained
{', '.join(self.config.stage_focus)}

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.config.output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{self.config.output_dir}")

# Generate code
prompt = "### Instruction: Implement a binary search tree\\n\\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
```
"""
        
        with open(f"{self.config.output_dir}/README.md", 'w') as f:
            f.write(model_card)

def main(
    training_data: str = "zen_coder_data/zen_coder_train.json",
    validation_data: str = "zen_coder_data/zen_coder_val.json",
    output_dir: str = "zen-coder-finetuned",
    use_bitdelta: bool = True,
    num_epochs: int = 3,
    **kwargs
):
    """
    Train Zen-Coder on software engineering patterns
    
    Args:
        training_data: Path to training JSON
        validation_data: Path to validation JSON
        output_dir: Output directory for model
        use_bitdelta: Enable BitDelta compression
        num_epochs: Number of training epochs
    """
    
    config = CodeTrainingConfig(
        training_data=training_data,
        validation_data=validation_data,
        output_dir=output_dir,
        use_bitdelta=use_bitdelta,
        num_epochs=num_epochs,
        **kwargs
    )
    
    trainer = ZenCoderTrainer(config)
    trainer.train()

if __name__ == "__main__":
    fire.Fire(main)