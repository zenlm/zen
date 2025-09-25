#!/bin/bash
# One-click Zen Nano training with Gym

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     🧘 Zen Nano Training with Gym                          ║"
echo "║     Choose your training method:                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "1) 🌐 Web UI (Recommended - Visual interface)"
echo "2) 🖥️  CLI Quick Train (Fast, no UI)"
echo "3) 🧪 Test existing model"
echo "4) 📦 Export to GGUF"
echo ""
read -p "Select option (1-4): " option

GYM_PATH="/Users/z/work/zoo/gym"
ZEN_NANO_PATH="/Users/z/work/zen/zen-nano"

case $option in
    1)
        echo "🚀 Launching Gym Web UI..."
        echo ""
        echo "📝 Instructions:"
        echo "   1. The UI will open in your browser"
        echo "   2. In the UI, select:"
        echo "      • Model: Qwen/Qwen3-4B-Instruct"
        echo "      • Dataset: zen_nano (will appear in dropdown)"
        echo "      • Template: qwen3"
        echo "      • Training Method: QLoRA"
        echo "   3. Click 'Start Training'"
        echo ""
        cd $GYM_PATH
        python -m llamafactory.webui.interface
        ;;
        
    2)
        echo "🏃 Starting Quick CLI Training..."
        cd $GYM_PATH
        python -m llamafactory.train \
            --model_name_or_path "Qwen/Qwen3-4B-Instruct" \
            --template "qwen3" \
            --dataset "zen_nano" \
            --dataset_dir "./data" \
            --cutoff_len 2048 \
            --stage "sft" \
            --do_train True \
            --finetuning_type "lora" \
            --quantization_bit 4 \
            --lora_rank 16 \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --lora_target "all" \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 8 \
            --learning_rate 1e-4 \
            --num_train_epochs 3 \
            --lr_scheduler_type "cosine" \
            --warmup_ratio 0.1 \
            --gradient_checkpointing True \
            --output_dir "./output/zen-nano" \
            --logging_steps 10 \
            --save_steps 100 \
            --save_total_limit 3 \
            --plot_loss True \
            --overwrite_output_dir True
        ;;
        
    3)
        echo "🧪 Testing Zen Nano model..."
        cd $GYM_PATH
        python -m llamafactory.chat \
            --model_name_or_path "Qwen/Qwen3-4B-Instruct" \
            --adapter_name_or_path "./output/zen-nano" \
            --template "qwen3"
        ;;
        
    4)
        echo "📦 Exporting to GGUF format..."
        echo "This will create a file you can use with llama.cpp or Ollama"
        cd $GYM_PATH
        python scripts/export_model.py \
            --model_name_or_path "Qwen/Qwen3-4B-Instruct" \
            --adapter_name_or_path "./output/zen-nano" \
            --export_dir "./exports" \
            --export_hub_model_id "zen-nano-4b-gguf" \
            --export_quantization_bit 4 \
            --export_quantization_dataset "zen_nano" \
            --export_legacy_format False
        echo "✅ Exported to: ./exports/zen-nano-4b-gguf"
        ;;
        
    *)
        echo "Invalid option. Please run the script again and select 1-4."
        ;;
esac