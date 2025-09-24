# Qwen3-Omni Fine-tuning Framework

Advanced fine-tuning framework for Qwen3-Omni-30B-A3B multimodal models supporting text, audio, image, and video inputs with text and speech generation.

## 🚀 Model Overview

**Qwen3-Omni-30B-A3B** is a natively end-to-end multilingual omni-modal foundation model featuring:

- **Multimodal Understanding**: Processes text, images, audio, and video
- **Dual Output**: Generates both text and natural speech in real-time
- **MoE Architecture**: Thinker-Talker design with 30B total / 3B active parameters
- **Multilingual**: 119 text languages, 19 speech input, 10 speech output languages
- **State-of-the-art**: SOTA on 22 of 36 audio/video benchmarks

## 📦 Model Variants

### 1. Qwen3-Omni-30B-A3B-Instruct
- **Components**: Thinker + Talker (full model)
- **Inputs**: Audio, Video, Text, Image
- **Outputs**: Text + Audio (speech synthesis)
- **Use Case**: Interactive voice assistants, multimodal chat

### 2. Qwen3-Omni-30B-A3B-Thinking
- **Components**: Thinker only
- **Inputs**: Audio, Video, Text, Image
- **Outputs**: Text only
- **Use Case**: Complex reasoning, chain-of-thought tasks

### 3. Qwen3-Omni-30B-A3B-Captioner
- **Components**: Fine-tuned Thinker
- **Inputs**: Audio
- **Outputs**: Text (detailed captions)
- **Use Case**: High-quality audio captioning

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│             Input Processing                 │
├─────────┬─────────┬─────────┬───────────────┤
│  Text   │  Audio  │  Image  │    Video      │
│ Encoder │ Encoder │ Encoder │   Encoder     │
└────┬────┴────┬────┴────┬────┴───────┬───────┘
     │         │         │            │
     ▼         ▼         ▼            ▼
┌─────────────────────────────────────────────┐
│          Thinker (MoE Transformer)          │
│        30B params / 3B active params        │
│         Chain-of-Thought Reasoning          │
└─────────────────────────┬───────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
     ┌───────────────┐     ┌───────────────┐
     │ Text Decoder  │     │ Audio Talker  │
     │  (Text Gen)   │     │ (Speech Gen)  │
     └───────────────┘     └───────────────┘
```

## 🛠️ Installation

```bash
# Install dependencies
pip install -U transformers accelerate flash-attn qwen-omni-utils
pip install git+https://github.com/huggingface/transformers
pip install soundfile librosa

# For fine-tuning
pip install peft bitsandbytes deepspeed
pip install wandb tensorboard

# Optional: Install vLLM for efficient inference
git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
cd vllm && pip install -e . -v
```

## 📊 Fine-tuning Setup

### Directory Structure
```
qwen3-omni/
├── configs/              # Training configurations
│   ├── instruct.yaml     # Instruction fine-tuning
│   ├── thinking.yaml     # CoT reasoning fine-tuning
│   └── captioner.yaml    # Audio captioning fine-tuning
├── data/                 # Training datasets
│   ├── multimodal/       # Mixed modality data
│   ├── audio/            # Audio-specific data
│   └── video/            # Video-specific data
├── scripts/              # Training scripts
│   ├── finetune.py       # Main fine-tuning script
│   ├── prepare_data.py   # Data preprocessing
│   └── evaluate.py       # Model evaluation
├── adapters/             # LoRA adapters
└── checkpoints/          # Saved models
```

## 🔧 Quick Start

### 1. Download Base Model
```python
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

model_name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
)
processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
```

### 2. Basic Inference
```python
from qwen_omni_utils import process_mm_info
import soundfile as sf

# Prepare multimodal input
conversation = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "path/to/image.jpg"},
        {"type": "audio", "audio": "path/to/audio.wav"},
        {"type": "text", "text": "What do you see and hear?"}
    ]
}]

# Process inputs
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
inputs = processor(
    text=text, audio=audios, images=images, videos=videos,
    return_tensors="pt", padding=True, use_audio_in_video=True
).to(model.device)

# Generate text and audio
text_ids, audio = model.generate(
    **inputs,
    speaker="Ethan",  # Voice selection
    thinker_return_dict_in_generate=True,
    use_audio_in_video=True
)

# Decode outputs
text_output = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1]:])
if audio is not None:
    sf.write("output.wav", audio.reshape(-1).numpy(), samplerate=24000)
```

## 🎯 Fine-tuning Tasks

### 1. Instruction Fine-tuning
- Improve following complex multimodal instructions
- Enhance response quality and formatting
- Optimize for specific domains (medical, legal, technical)

### 2. Chain-of-Thought Fine-tuning
- Enhance reasoning capabilities
- Improve step-by-step problem solving
- Better mathematical and logical reasoning

### 3. Audio Captioning
- Generate detailed audio descriptions
- Reduce hallucinations
- Domain-specific audio understanding

### 4. Voice Cloning/Adaptation
- Customize speech output characteristics
- Train on specific speaker profiles
- Emotional expression enhancement

## 📈 Training Configuration

### LoRA Configuration for MoE
```yaml
lora:
  rank: 64              # Higher rank for MoE
  alpha: 128
  dropout: 0.1
  target_modules:       # Target MoE expert layers
    - gate
    - experts.*.wi
    - experts.*.wo

moe:
  num_experts: 8
  num_experts_per_tok: 2
  expert_capacity: 1.25
```

### Multimodal Training
```yaml
training:
  batch_size: 1         # Large model requires small batch
  gradient_accumulation: 16
  learning_rate: 2e-5
  warmup_ratio: 0.03
  epochs: 3

  # Modality weights
  loss_weights:
    text: 1.0
    audio: 0.8
    vision: 0.9
```

## 💾 Memory Requirements

| Model | Precision | Video Length | Min GPU Memory |
|-------|-----------|--------------|----------------|
| Instruct | BF16 | 15s | 78.85 GB |
| Instruct | BF16 | 30s | 88.52 GB |
| Instruct | BF16 | 60s | 107.74 GB |
| Thinking | BF16 | 15s | 68.74 GB |
| Thinking | BF16 | 30s | 77.79 GB |
| Thinking | BF16 | 60s | 95.76 GB |

### Optimization Strategies
- Use 4-bit/8-bit quantization
- Gradient checkpointing
- DeepSpeed ZeRO-3
- Multi-GPU with tensor parallelism

## 🌍 Language Support

### Speech Input (19 languages)
English, Chinese, Korean, Japanese, German, Russian, Italian, French, Spanish, Portuguese, Malay, Dutch, Indonesian, Turkish, Vietnamese, Cantonese, Arabic, Urdu

### Speech Output (10 languages)
English, Chinese, French, German, Russian, Italian, Spanish, Portuguese, Japanese, Korean

### Text (119 languages)
Full support for 119 text languages including all major world languages

## 📊 Performance Benchmarks

| Benchmark | Score | Notes |
|-----------|-------|-------|
| ASR (LibriSpeech) | 2.1 WER | Near SOTA |
| MMLU | 74.2 | Strong reasoning |
| Audio Understanding | 89.3 | SOTA on 22/36 tasks |
| Video QA | 81.5 | Competitive with GPT-4V |

## 🔬 Advanced Features

### 1. Real-time Streaming
- Low-latency audio generation
- Natural turn-taking in conversations
- Streaming text and audio outputs

### 2. System Prompt Customization
```python
system_prompt = """You are Qwen-Omni, a smart voice assistant.
Keep responses brief (under 50 words).
Use natural, conversational language.
Interact as if talking face-to-face."""
```

### 3. Multi-codebook Design
- Efficient audio encoding
- Minimal latency
- High-quality speech synthesis

## 🚀 Deployment

### vLLM Deployment (Recommended)
```python
from vllm import LLM
llm = LLM(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    tensor_parallel_size=4,  # Multi-GPU
    gpu_memory_utilization=0.95,
    limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 3}
)
```

### Docker Deployment
```dockerfile
FROM nvidia/cuda:12.1-runtime
# Coming soon: Official Qwen3-Omni container
```

## 📝 Data Format

### Multimodal Conversation Format
```json
{
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "audio", "audio": "audio_path.wav"},
        {"type": "image", "image": "image_path.jpg"},
        {"type": "video", "video": "video_path.mp4"},
        {"type": "text", "text": "Describe what you perceive"}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "I can see and hear..."}]
    }
  ]
}
```

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Domain-specific fine-tuning recipes
- Efficient training techniques for large MoE models
- Multimodal dataset creation tools
- Voice synthesis improvements

## 📄 License

Apache 2.0 - Model weights and code

## 🙏 Acknowledgments

- Qwen team for the groundbreaking Qwen3-Omni model
- Hugging Face for transformers integration
- vLLM team for efficient inference support

## 📚 Citation

```bibtex
@article{qwen3-omni,
  title={Qwen3-Omni: Natively Multimodal Foundation Model},
  author={Qwen Team},
  year={2024}
}
```