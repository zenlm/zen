"""
Training utilities for Zen models
Common functions for training setup, monitoring, and optimization
"""

import os
import torch
import wandb
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
from transformers import TrainerCallback, TrainerState, TrainerControl
import psutil
import GPUtil


class ZenTrainingMonitor(TrainerCallback):
    """Custom callback for monitoring Zen model training"""
    
    def __init__(self, log_dir: str = "./logs", use_wandb: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.training_history = []
        
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, Any], **kwargs):
        """Log training metrics"""
        # Add timestamp
        logs['timestamp'] = datetime.now().isoformat()
        logs['step'] = state.global_step
        logs['epoch'] = state.epoch
        
        # Add system metrics
        logs.update(self._get_system_metrics())
        
        # Store in history
        self.training_history.append(logs)
        
        # Log to wandb if configured
        if self.use_wandb:
            wandb.log(logs)
        
        # Save to file
        log_file = self.log_dir / f"training_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(logs) + '\n')
    
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Save training history on checkpoint"""
        history_file = Path(args.output_dir) / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics"""
        metrics = {}
        
        # CPU metrics
        metrics['cpu_percent'] = psutil.cpu_percent()
        metrics['memory_percent'] = psutil.virtual_memory().percent
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                metrics['gpu_utilization'] = gpu.load * 100
                metrics['gpu_memory_used'] = gpu.memoryUsed
                metrics['gpu_memory_total'] = gpu.memoryTotal
                metrics['gpu_temperature'] = gpu.temperature
        except:
            pass
        
        return metrics


class OptimizationUtils:
    """Utilities for training optimization"""
    
    @staticmethod
    def setup_mixed_precision(fp16: bool = False, bf16: bool = False) -> Dict[str, Any]:
        """Setup mixed precision training"""
        config = {}
        
        if fp16:
            config['fp16'] = True
            config['half_precision_backend'] = 'auto'
        elif bf16:
            config['bf16'] = True
            
        return config
    
    @staticmethod
    def calculate_gradient_accumulation(
        batch_size: int,
        target_batch_size: int,
        world_size: int = 1
    ) -> int:
        """Calculate gradient accumulation steps"""
        effective_batch_size = batch_size * world_size
        accumulation_steps = max(1, target_batch_size // effective_batch_size)
        return accumulation_steps
    
    @staticmethod
    def get_linear_schedule_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1
    ):
        """Create linear schedule with warmup"""
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch
        )
    
    @staticmethod
    def setup_distributed_training() -> Dict[str, Any]:
        """Setup distributed training configuration"""
        config = {}
        
        if torch.cuda.device_count() > 1:
            config['ddp'] = True
            config['ddp_find_unused_parameters'] = False
            config['dataloader_num_workers'] = 4
            config['local_rank'] = int(os.environ.get('LOCAL_RANK', -1))
        
        return config


class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        
    def save_checkpoint(
        self,
        model,
        tokenizer,
        optimizer,
        scheduler,
        epoch: int,
        step: int,
        metrics: Dict[str, Any]
    ) -> str:
        """Save training checkpoint"""
        checkpoint_name = f"checkpoint-epoch{epoch}-step{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(str(checkpoint_path))
        tokenizer.save_pretrained(str(checkpoint_path))
        
        # Save optimizer and scheduler
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'step': step,
            'metrics': metrics
        }, checkpoint_path / "training_state.pt")
        
        # Save metrics
        with open(checkpoint_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model,
        optimizer=None,
        scheduler=None
    ) -> Dict[str, Any]:
        """Load training checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        # Load training state
        training_state = torch.load(checkpoint_path / "training_state.pt")
        
        if optimizer:
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
        
        if scheduler and training_state.get('scheduler_state_dict'):
            scheduler.load_state_dict(training_state['scheduler_state_dict'])
        
        return {
            'epoch': training_state['epoch'],
            'step': training_state['step'],
            'metrics': training_state['metrics']
        }
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the last N"""
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for checkpoint in checkpoints[self.keep_last_n:]:
            import shutil
            shutil.rmtree(checkpoint)


def setup_training(
    model_name: str,
    training_stage: str,
    output_dir: str,
    use_wandb: bool = False,
    project_name: str = "zen-training"
) -> Dict[str, Any]:
    """Setup training environment and configuration"""
    
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if configured
    if use_wandb:
        wandb.init(
            project=project_name,
            name=f"{model_name}-{training_stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "model": model_name,
                "stage": training_stage,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # Setup configuration
    config = {
        "model_name": model_name,
        "training_stage": training_stage,
        "output_dir": str(output_dir),
        "log_dir": str(log_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "timestamp": datetime.now().isoformat(),
        "use_wandb": use_wandb
    }
    
    # Add optimization settings
    config.update(OptimizationUtils.setup_distributed_training())
    
    # Save initial config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return config


def calculate_training_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """Calculate training metrics"""
    metrics = {}
    
    # Calculate accuracy
    if predictions.dim() > 1:
        pred_labels = torch.argmax(predictions, dim=-1)
    else:
        pred_labels = predictions
    
    correct = (pred_labels == labels).float()
    metrics['accuracy'] = correct.mean().item()
    
    # Calculate perplexity
    if predictions.dim() > 2:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fn(predictions.view(-1, predictions.size(-1)), labels.view(-1))
        metrics['perplexity'] = torch.exp(loss).item()
    
    return metrics


def save_training_summary(
    output_dir: str,
    model_name: str,
    training_config: Dict[str, Any],
    final_metrics: Dict[str, Any],
    training_time: float
):
    """Save comprehensive training summary"""
    
    summary = {
        "model": model_name,
        "configuration": training_config,
        "final_metrics": final_metrics,
        "training_time_hours": training_time / 3600,
        "completed_at": datetime.now().isoformat()
    }
    
    summary_path = Path(output_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Also create a markdown summary
    md_summary = f"""
# Training Summary: {model_name}

## Configuration
- **Model**: {model_name}
- **Stage**: {training_config.get('training_stage', 'unknown')}
- **Epochs**: {training_config.get('num_epochs', 'unknown')}
- **Batch Size**: {training_config.get('batch_size', 'unknown')}
- **Learning Rate**: {training_config.get('learning_rate', 'unknown')}

## Final Metrics
"""
    
    for key, value in final_metrics.items():
        md_summary += f"- **{key}**: {value:.4f}\n"
    
    md_summary += f"\n## Training Time\n- **Total**: {training_time/3600:.2f} hours\n"
    md_summary += f"\n## Completed\n- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    md_path = Path(output_dir) / "README.md"
    with open(md_path, "w") as f:
        f.write(md_summary)
    
    print(f"Training summary saved to {summary_path}")


def estimate_memory_usage(
    model_size_gb: float,
    batch_size: int,
    sequence_length: int,
    use_gradient_checkpointing: bool = False
) -> Dict[str, float]:
    """Estimate GPU memory usage"""
    
    # Base model memory
    model_memory = model_size_gb
    
    # Gradient memory (roughly same as model)
    gradient_memory = model_size_gb
    
    # Optimizer states (AdamW uses 2x model size)
    optimizer_memory = model_size_gb * 2
    
    # Activation memory (rough estimate)
    activation_memory = (batch_size * sequence_length * 4096 * 4) / (1024**3)
    
    if use_gradient_checkpointing:
        activation_memory *= 0.3  # Reduce by ~70% with checkpointing
    
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        "model_memory_gb": model_memory,
        "gradient_memory_gb": gradient_memory,
        "optimizer_memory_gb": optimizer_memory,
        "activation_memory_gb": activation_memory,
        "total_memory_gb": total_memory,
        "recommended_gpu": _recommend_gpu(total_memory)
    }


def _recommend_gpu(memory_gb: float) -> str:
    """Recommend GPU based on memory requirements"""
    if memory_gb <= 16:
        return "RTX 4070 (16GB) or better"
    elif memory_gb <= 24:
        return "RTX 4090 (24GB) or better"
    elif memory_gb <= 40:
        return "A100 40GB or better"
    elif memory_gb <= 80:
        return "A100 80GB or better"
    else:
        return "Multiple A100 80GB with model parallelism"