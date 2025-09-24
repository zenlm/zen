"""
Model configurations for all Zen variants
Maps model names to their base models and optimal training parameters
"""

MODEL_CONFIGS = {
    "zen-nano": {
        "base_model": "Qwen/Qwen3-4B",
        "model_type": "causal_lm",
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_hidden_layers": 32,
        "training_params": {
            "learning_rate": 2e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "warmup_ratio": 0.1,
            "fp16": True,
        }
    },
    
    "zen-nano-instruct": {
        "base_model": "Qwen/Qwen3-4B-Instruct",
        "model_type": "causal_lm",
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_hidden_layers": 32,
        "training_params": {
            "learning_rate": 1e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 4096,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "warmup_ratio": 0.05,
            "fp16": True,
        }
    },
    
    "zen-nano-thinking": {
        "base_model": "Qwen/Qwen3-4B-Thinking",
        "model_type": "causal_lm",
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_hidden_layers": 32,
        "training_params": {
            "learning_rate": 1e-4,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 8192,  # Longer for thinking traces
            "lora_r": 64,
            "lora_alpha": 128,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "warmup_ratio": 0.1,
            "fp16": True,
            "gradient_checkpointing": True,
        }
    },
    
    "zen-omni": {
        "base_model": "Qwen/Qwen3-Omni-30B-A3B",
        "model_type": "multimodal",
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_hidden_layers": 60,
        "num_experts": 8,
        "num_experts_per_token": 2,
        "training_params": {
            "learning_rate": 5e-5,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_seq_length": 2048,
            "lora_r": 128,
            "lora_alpha": 256,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up_proj", "down_proj"],
            "warmup_ratio": 0.1,
            "bf16": True,  # Use bf16 for larger models
            "gradient_checkpointing": True,
            "use_flash_attention": True,
        }
    },
    
    "zen-omni-instruct": {
        "base_model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "model_type": "multimodal",
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_hidden_layers": 60,
        "num_experts": 8,
        "num_experts_per_token": 2,
        "training_params": {
            "learning_rate": 2e-5,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_seq_length": 4096,
            "lora_r": 128,
            "lora_alpha": 256,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up_proj", "down_proj"],
            "warmup_ratio": 0.05,
            "bf16": True,
            "gradient_checkpointing": True,
            "use_flash_attention": True,
        }
    },
    
    "zen-omni-thinking": {
        "base_model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "model_type": "multimodal",
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_hidden_layers": 60,
        "num_experts": 8,
        "num_experts_per_token": 2,
        "training_params": {
            "learning_rate": 1e-5,
            "batch_size": 1,
            "gradient_accumulation_steps": 32,
            "max_seq_length": 8192,
            "lora_r": 256,
            "lora_alpha": 512,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up_proj", "down_proj"],
            "warmup_ratio": 0.1,
            "bf16": True,
            "gradient_checkpointing": True,
            "use_flash_attention": True,
        }
    },
    
    "zen-omni-captioner": {
        "base_model": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
        "model_type": "multimodal",
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_hidden_layers": 60,
        "training_params": {
            "learning_rate": 2e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 1024,
            "lora_r": 64,
            "lora_alpha": 128,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "warmup_ratio": 0.1,
            "bf16": True,
            "gradient_checkpointing": True,
        }
    },
    
    "zen-coder": {
        "base_model": "Qwen/Qwen3-4B",
        "model_type": "causal_lm",
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_hidden_layers": 32,
        "training_params": {
            "learning_rate": 2e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 4096,
            "lora_r": 64,
            "lora_alpha": 128,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "warmup_ratio": 0.1,
            "fp16": True,
            "gradient_checkpointing": True,
        }
    },
    
    "zen-next": {
        "base_model": "Qwen/Qwen3-4B",
        "model_type": "causal_lm",
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_hidden_layers": 32,
        "experimental": True,
        "training_params": {
            "learning_rate": 3e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "warmup_ratio": 0.15,
            "fp16": True,
            "use_bitdelta": True,
            "bitdelta_ratio": 0.15,
        }
    },
}

# QLoRA configurations for memory-efficient training
QLORA_CONFIGS = {
    "zen-nano": {
        "bits": 4,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    },
    "zen-omni": {
        "bits": 4,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    },
    "zen-coder": {
        "bits": 8,
        "bnb_8bit_compute_dtype": "float16",
    }
}

# DeepSpeed configurations for distributed training
DEEPSPEED_CONFIGS = {
    "stage2": {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    },
    "stage3": {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }
}


def get_model_config(model_name: str) -> dict:
    """Get configuration for a specific model variant"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]


def get_qlora_config(model_name: str) -> dict:
    """Get QLoRA configuration for a specific model"""
    return QLORA_CONFIGS.get(model_name, QLORA_CONFIGS["zen-nano"])


def get_deepspeed_config(stage: str = "stage2") -> dict:
    """Get DeepSpeed configuration"""
    return DEEPSPEED_CONFIGS.get(stage, DEEPSPEED_CONFIGS["stage2"])


def list_available_models() -> list:
    """List all available model variants"""
    return list(MODEL_CONFIGS.keys())


def get_optimal_batch_size(model_name: str, gpu_memory: int) -> int:
    """Calculate optimal batch size based on model and GPU memory"""
    config = get_model_config(model_name)
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    
    # Rough estimation of memory usage
    model_size_gb = (hidden_size * num_layers * 4) / (1024**3)  # Approximate
    
    if gpu_memory >= 80:  # A100 80GB
        return 8 if "omni" in model_name else 16
    elif gpu_memory >= 40:  # A100 40GB
        return 4 if "omni" in model_name else 8
    elif gpu_memory >= 24:  # RTX 4090
        return 2 if "omni" in model_name else 4
    elif gpu_memory >= 16:  # RTX 4070
        return 1 if "omni" in model_name else 2
    else:
        return 1