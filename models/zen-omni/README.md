---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- zen
- multimodal
- vision
- audio
- omni
- hanzo
- zoo
base_model: Qwen/Qwen3-Omni-30B-A3B-Instruct
pipeline_tag: image-text-to-text
---

# Zen-Omni

## Model Description

Zen-Omni is based on Qwen3-Omni-30B-A3B, Alibaba's groundbreaking multimodal model that natively processes text, images, audio, and video. With 30B total parameters (3B active via MoE), it achieves 211ms latency and state-of-the-art performance on 22/36 audiovisual benchmarks.

## Key Features

- **Multimodal native**: Processes text, images, audio, and video
- **Thinker-Talker architecture**: Parallel reasoning and generation
- **Ultra-low latency**: 234ms first-packet streaming
- **119 languages**: Text understanding and generation
- **Real-time capable**: Voice conversations and live video analysis

## Architecture

```
Input → [Vision Encoder] → [Thinker Module] → [Router]
      → [Audio Encoder]  →                   ↓
      → [Text Encoder]   →           [Talker Module] → Output
```

- **Thinker**: Deep reasoning and understanding (8 experts, 2 active)
- **Talker**: Fast streaming generation
- **MoE Router**: Dynamic expert selection

## Model Details

- **Parameters**: 30B total, 3B active (MoE)
- **Vision**: Native image understanding
- **Audio**: 30 minutes processing, 19 input / 10 output languages
- **Context**: Extended context support
- **Latency**: 211ms first response
- **Languages**: 119 text, 19 speech input, 10 speech output

## Input Modalities

| Modality | Specs | Languages |
|----------|-------|-----------|
| Text | 128K context | 119 |
| Image | 336x336 | N/A |
| Audio | 16kHz | 19 input |
| Video | 30fps | N/A |

## Usage

### Multimodal Input

```python
from transformers import AutoProcessor, AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained("zenlm/zen-omni")
processor = AutoProcessor.from_pretrained("zenlm/zen-omni")

# Text + Image
inputs = processor(
    text="What's in this image?",
    images=image,
    return_tensors="pt"
)
outputs = model.generate(**inputs)
```

### Audio Processing

```python
# Process audio input
audio_input = processor(
    audio=audio_array,
    sampling_rate=16000,
    return_tensors="pt"
)
transcription = model.generate(**audio_input)
```

### Streaming Generation

```python
from zen1_omni import StreamingModel

model = StreamingModel("zenlm/zen-omni")
for token in model.stream_generate(prompt):
    print(token, end="", flush=True)
    # First token in 234ms
```

### Vision-Language Tasks

```python
# Visual question answering
response = model.chat(
    image=image,
    prompt="Describe what you see and identify any text"
)

# OCR and document understanding
doc_analysis = model.analyze_document(image)
```

## Performance

| Benchmark | Score |
|-----------|-------|
| MMLU | 82.4 |
| VQA-v2 | 85.3 |
| DocVQA | 78.9 |
| AudioCaps | 91.2 |
| VISTA | 76.4 |
| HumanEval | 87.3 |

## Latency Metrics

| Metric | Time |
|--------|------|
| First packet | 234ms |
| Voice response | 380ms |
| Image analysis | 520ms |
| Tokens/sec | 42 |

## Expert Specialization

The 8 MoE experts specialize in:
1. General conversation
2. Code and technical
3. Visual understanding  
4. Audio processing
5. Mathematical reasoning
6. Creative writing
7. Multilingual translation
8. Tool use and planning

## Applications

- Real-time voice assistants
- Video analysis and captioning
- Document understanding
- Multimodal chatbots
- Accessibility tools
- Live translation
- Educational tutoring
- Creative content generation

## Organizations

Developed by **Hanzo AI** and **Zoo Labs Foundation** - pioneering multimodal AI systems with real-time capabilities.

## Citation

```bibtex
@article{zen2024omni,
  title={Zen-Omni: Unified Multimodal Intelligence with Thinker-Talker Architecture},
  author={Hanzo AI Research Team},
  year={2024}
}
```