#!/bin/bash
# Zen Nano Training Launcher for Gym

echo "🧘 Zen Nano Training with Gym UI"
echo "================================"

cd /Users/z/work/zoo/gym

# Launch Gym's Web UI
echo "🌐 Launching Gym Web UI..."
echo "📊 Select 'zen_nano' from the dataset dropdown"
echo "🤖 Model: Qwen3-4B-Instruct"
echo ""

python -m llamafactory.webui.interface
