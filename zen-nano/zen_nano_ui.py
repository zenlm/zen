#!/usr/bin/env python3
"""
Zen Nano Training UI - Beautiful Gradio Interface
Powered by Gym Platform
Jointly developed by Hanzo AI Inc & Zoo Labs Foundation
"""

import os
import sys
import json
import gradio as gr
from pathlib import Path
from datetime import datetime
import subprocess
import threading
import queue
import time

# Add Gym to path
GYM_PATH = Path("/Users/z/work/zoo/gym")
if GYM_PATH.exists():
    sys.path.insert(0, str(GYM_PATH / "src"))

class ZenNanoTrainer:
    def __init__(self):
        self.training_process = None
        self.log_queue = queue.Queue()
        self.is_training = False
        self.training_logs = []
        
    def prepare_training_config(self, model_size, method, dataset, 
                              batch_size, learning_rate, epochs, 
                              lora_rank, gradient_accumulation):
        """Prepare Gym training configuration"""
        
        model_map = {
            "Qwen3-4B (Recommended)": "Qwen/Qwen3-4B-Instruct",
            "Qwen2.5-1.5B (Faster)": "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen2.5-0.5B (Testing)": "Qwen/Qwen2.5-0.5B-Instruct",
        }
        
        config = {
            "model_name_or_path": model_map[model_size],
            "template": "qwen3" if "Qwen3" in model_size else "qwen",
            "dataset": dataset,
            "dataset_dir": "./training",
            "cutoff_len": 2048,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "learning_rate": learning_rate,
            "num_train_epochs": epochs,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "gradient_checkpointing": True,
            "output_dir": f"./gym-output/zen-nano-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "logging_steps": 10,
            "save_steps": 100,
            "save_total_limit": 3,
            "overwrite_output_dir": True,
        }
        
        if method == "QLoRA (4-bit, Low Memory)":
            config.update({
                "quantization_bit": 4,
                "bnb_4bit_compute_dtype": "float32",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "lora_rank": lora_rank,
                "lora_alpha": lora_rank * 2,
                "lora_dropout": 0.05,
                "lora_target": "all",
            })
        elif method == "LoRA (Standard)":
            config.update({
                "lora_rank": lora_rank,
                "lora_alpha": lora_rank * 2,
                "lora_dropout": 0.1,
                "lora_target": "all",
            })
        else:  # Full
            config["finetuning_type"] = "full"
        
        return config
    
    def start_training(self, config):
        """Start training in background thread"""
        self.is_training = True
        self.training_logs = []
        
        # Save config
        config_path = Path(config["output_dir"]) / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Start training thread
        def train():
            try:
                from llamafactory.train import run_sft
                from llamafactory.hparams import get_train_args
                
                self.log_queue.put("🚀 Starting training...")
                self.log_queue.put(f"📁 Output: {config['output_dir']}")
                
                # Run training
                model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config)
                run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
                
                self.log_queue.put("✅ Training completed successfully!")
                
            except Exception as e:
                self.log_queue.put(f"❌ Error: {str(e)}")
            finally:
                self.is_training = False
        
        thread = threading.Thread(target=train, daemon=True)
        thread.start()
        
    def get_logs(self):
        """Get training logs"""
        new_logs = []
        while not self.log_queue.empty():
            try:
                log = self.log_queue.get_nowait()
                new_logs.append(log)
                self.training_logs.append(log)
            except queue.Empty:
                break
        
        return "\n".join(self.training_logs[-100:])  # Keep last 100 lines

# Initialize trainer
trainer = ZenNanoTrainer()

def create_ui():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Zen Nano Training UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧘 Zen Nano v1.0 Training Interface
        ### Powered by Gym Platform | By Hanzo AI Inc & Zoo Labs Foundation
        
        Train your own Zen Nano model with this intuitive interface. 
        Zen Nano is an ultra-lightweight AI model optimized for edge deployment.
        """)
        
        with gr.Tabs():
            # Training Tab
            with gr.TabItem("🏋️ Training"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Model Configuration")
                        
                        model_size = gr.Dropdown(
                            label="Base Model",
                            choices=["Qwen3-4B (Recommended)", 
                                   "Qwen2.5-1.5B (Faster)",
                                   "Qwen2.5-0.5B (Testing)"],
                            value="Qwen3-4B (Recommended)"
                        )
                        
                        method = gr.Dropdown(
                            label="Training Method",
                            choices=["QLoRA (4-bit, Low Memory)", 
                                   "LoRA (Standard)",
                                   "Full Fine-tuning (High Memory)"],
                            value="QLoRA (4-bit, Low Memory)"
                        )
                        
                        dataset = gr.Dropdown(
                            label="Dataset",
                            choices=["zen_nano", 
                                   "zen_nano_identity",
                                   "custom"],
                            value="zen_nano"
                        )
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### Training Parameters")
                        
                        with gr.Row():
                            batch_size = gr.Slider(
                                label="Batch Size",
                                minimum=1, maximum=16, value=2, step=1
                            )
                            
                            gradient_accumulation = gr.Slider(
                                label="Gradient Accumulation",
                                minimum=1, maximum=32, value=8, step=1
                            )
                        
                        with gr.Row():
                            learning_rate = gr.Number(
                                label="Learning Rate",
                                value=1e-4,
                                minimum=1e-6,
                                maximum=1e-2
                            )
                            
                            epochs = gr.Slider(
                                label="Epochs",
                                minimum=1, maximum=10, value=3, step=1
                            )
                        
                        lora_rank = gr.Slider(
                            label="LoRA Rank",
                            minimum=4, maximum=64, value=16, step=4,
                            visible=True
                        )
                
                gr.Markdown("### Training Control")
                
                with gr.Row():
                    train_btn = gr.Button("🚀 Start Training", variant="primary", scale=2)
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)
                    export_btn = gr.Button("📦 Export Model", scale=1)
                
                # Training logs
                gr.Markdown("### Training Progress")
                training_logs = gr.Textbox(
                    label="Training Logs",
                    lines=15,
                    max_lines=20,
                    autoscroll=True,
                    interactive=False
                )
                
                # Progress bar
                progress = gr.Progress()
                
            # Dataset Tab
            with gr.TabItem("📊 Dataset"):
                gr.Markdown("""
                ### Dataset Management
                Upload your own training data or use pre-configured datasets.
                """)
                
                with gr.Row():
                    with gr.Column():
                        dataset_upload = gr.File(
                            label="Upload Dataset (JSONL)",
                            file_types=[".jsonl", ".json"],
                            type="filepath"
                        )
                        
                        dataset_preview = gr.Dataframe(
                            label="Dataset Preview",
                            headers=["Instruction", "Input", "Output"],
                            max_rows=10
                        )
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### Dataset Format
                        ```json
                        {
                            "instruction": "What is your name?",
                            "input": "",
                            "output": "I am Zen Nano v1.0..."
                        }
                        ```
                        
                        ### Available Datasets
                        - **zen_nano**: Core identity training (48 examples)
                        - **zen_nano_identity**: Extended identity (100+ examples)
                        - **custom**: Upload your own dataset
                        """)
            
            # Testing Tab
            with gr.TabItem("🧪 Testing"):
                gr.Markdown("### Test Your Fine-tuned Model")
                
                with gr.Row():
                    model_path = gr.Textbox(
                        label="Model Path",
                        placeholder="./gym-output/zen-nano-adapters",
                        value="./gym-output/zen-nano-adapters"
                    )
                    
                    load_model_btn = gr.Button("📥 Load Model")
                
                test_input = gr.Textbox(
                    label="Test Input",
                    placeholder="Ask me anything...",
                    lines=3
                )
                
                test_output = gr.Textbox(
                    label="Model Response",
                    lines=5,
                    interactive=False
                )
                
                test_btn = gr.Button("🔮 Generate Response", variant="primary")
            
            # Export Tab
            with gr.TabItem("📦 Export"):
                gr.Markdown("""
                ### Export Options
                Export your trained model in various formats.
                """)
                
                export_format = gr.Radio(
                    label="Export Format",
                    choices=["GGUF (llama.cpp)", "ONNX", "CoreML", "TensorRT", "Ollama"],
                    value="GGUF (llama.cpp)"
                )
                
                quantization = gr.Dropdown(
                    label="Quantization",
                    choices=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
                    value="Q4_K_M"
                )
                
                export_path = gr.Textbox(
                    label="Export Path",
                    value="./exports/zen-nano.gguf"
                )
                
                export_model_btn = gr.Button("📤 Export Model", variant="primary")
                export_status = gr.Textbox(label="Export Status", interactive=False)
            
            # About Tab
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About Zen Nano
                
                **Zen Nano v1.0** is an ultra-lightweight AI model jointly developed by:
                - **Hanzo AI Inc** - Techstars-backed applied AI research lab
                - **Zoo Labs Foundation** - 501(c)(3) non-profit organization
                
                ### Key Features
                - 🚀 **Edge Computing**: Runs entirely on local devices
                - 🔒 **Privacy First**: All data stays local
                - 🌱 **Eco-Friendly**: Minimal carbon footprint
                - 🎯 **Optimized**: 4B parameters, quantization ready
                - 🛠️ **Open Source**: Apache 2.0 license
                
                ### Training Infrastructure
                Powered by **Gym Platform** - The open-source AI training platform by Zoo Labs Foundation.
                
                ### Resources
                - 🌐 Website: [zoo.ngo](https://zoo.ngo)
                - 📚 Documentation: [docs.zoo.ngo/gym](https://docs.zoo.ngo/gym)
                - 💬 Discord: [discord.gg/zooai](https://discord.gg/zooai)
                - 📧 Contact: dev@zoo.ngo
                
                ### Credits
                Special thanks to the open-source AI community and all contributors.
                """)
        
        # Event handlers
        def start_training(model_size, method, dataset, batch_size, 
                         learning_rate, epochs, lora_rank, gradient_accumulation):
            """Start training handler"""
            config = trainer.prepare_training_config(
                model_size, method, dataset, batch_size,
                learning_rate, epochs, lora_rank, gradient_accumulation
            )
            
            trainer.start_training(config)
            
            # Update logs periodically
            logs = []
            while trainer.is_training:
                logs = trainer.get_logs()
                yield logs
                time.sleep(1)
            
            final_logs = trainer.get_logs()
            yield final_logs
        
        def load_dataset(file_path):
            """Load and preview dataset"""
            if file_path:
                try:
                    import pandas as pd
                    data = []
                    with open(file_path, "r") as f:
                        for line in f:
                            if line.strip():
                                item = json.loads(line)
                                data.append([
                                    item.get("instruction", ""),
                                    item.get("input", ""),
                                    item.get("output", "")[:100] + "..."
                                ])
                    
                    df = pd.DataFrame(data, columns=["Instruction", "Input", "Output"])
                    return df.head(10)
                except Exception as e:
                    return pd.DataFrame({"Error": [str(e)]})
            return None
        
        # Connect events
        train_btn.click(
            fn=start_training,
            inputs=[model_size, method, dataset, batch_size, 
                   learning_rate, epochs, lora_rank, gradient_accumulation],
            outputs=[training_logs]
        )
        
        dataset_upload.upload(
            fn=load_dataset,
            inputs=[dataset_upload],
            outputs=[dataset_preview]
        )
        
        # Update visibility based on method
        def update_lora_visibility(method):
            return gr.update(visible="LoRA" in method)
        
        method.change(
            fn=update_lora_visibility,
            inputs=[method],
            outputs=[lora_rank]
        )
    
    return demo

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║     🧘 Zen Nano v1.0 - Training UI                        ║
║     Powered by Gym Platform                               ║
║     Jointly by Hanzo AI Inc & Zoo Labs Foundation         ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    print("\n🌐 Starting web interface...")
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )