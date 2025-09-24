# Zen-Omni-Captioner

A specialized variant of the Zen-Omni multimodal foundation model, optimized for high-quality audio and video captioning with temporal alignment.

## Model Details

### Model Description

Zen-Omni-Captioner is a 30B parameter multimodal model with 3B active parameters using Mixture of Experts (MoE). This variant uses a balanced 50/50 weight distribution between Thinker and Talker modules, optimized for understanding and describing temporal media content.

- **Developed by:** Zen Research Team
- **Model type:** Multimodal Foundation Model (Thinker-Talker Architecture)
- **Language(s):** Multilingual (100+ languages)
- **License:** Apache 2.0
- **Modalities:** Text, Image, Audio, Video (specialized for temporal media)
- **Model Architecture:** Transformer with MoE and dual Thinker-Talker modules
- **Parameters:** 30B total, 3B active
- **Context Length:** 24,576 tokens
- **Processing:** 30 FPS video, 16kHz audio

### Model Sources

- **Repository:** [github.com/zenlm/zen-omni](https://github.com/zenlm/zen-omni)
- **Paper:** [Zen-Omni: A Thinker-Talker Architecture for Ultra-Low Latency Multimodal Understanding](https://arxiv.org/zen-omni)
- **Demo:** [zen-omni.zenlm.ai](https://zen-omni.zenlm.ai)

## Uses

### Direct Use

- Video captioning and description
- Audio transcription and analysis
- Live stream narration
- Accessibility services (audio descriptions)
- Content moderation and analysis
- Multimedia summarization
- Temporal event detection

### Downstream Use

The model can be fine-tuned for:
- Medical imaging description
- Security and surveillance analysis
- Sports commentary generation
- Educational content creation
- Film and media analysis
- Meeting transcription and summarization

### Out-of-Scope Use

- Pure text generation without media input
- Complex mathematical reasoning
- Tasks not involving temporal media

## Bias, Risks, and Limitations

### Limitations
- Optimized specifically for temporal media tasks
- May struggle with abstract reasoning without visual context
- Requires synchronized audio-video input for best performance

### Risks
- May miss subtle visual or auditory cues
- Potential biases in describing people and activities
- Risk of generating inappropriate descriptions for sensitive content

### Recommendations
- Validate captions for accuracy in critical applications
- Implement content filtering for sensitive domains
- Use human review for accessibility applications
- Consider cultural context in descriptions

## How to Get Started

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from zen_omni import VideoProcessor, AudioProcessor

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "zenlm/zen-omni-captioner",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-omni-captioner")

# Video captioning example
video_processor = VideoProcessor()
video_features = video_processor.process("path/to/video.mp4", fps=3)

prompt = "Describe what is happening in this video:"
inputs = tokenizer(prompt, return_tensors="pt")
inputs['video_features'] = video_features

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    num_beams=3  # Beam search for better captions
)

caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(caption)
```

### Temporal Alignment

```python
# Generate time-aligned captions
from zen_omni import TemporalCaptioner

captioner = TemporalCaptioner("zenlm/zen-omni-captioner")

# Process video with temporal alignment
results = captioner.caption_with_timestamps(
    video_path="video.mp4",
    segment_duration=5.0,  # 5-second segments
    overlap=1.0  # 1-second overlap
)

for segment in results:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['caption']}")
```

### Audio Captioning

```python
# Audio description generation
audio_processor = AudioProcessor()
audio_features = audio_processor.process("path/to/audio.wav")

prompt = "Describe the sounds in this audio:"
inputs = tokenizer(prompt, return_tensors="pt")
inputs['audio_features'] = audio_features

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.6,
    do_sample=True
)

description = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(description)
```

## Training Details

### Training Data

- **Video Datasets:** WebVid-10M, HowTo100M, YouTube-8M (150B tokens)
- **Audio Datasets:** AudioSet, LibriSpeech, CommonVoice (50B tokens)
- **Image-Caption:** COCO, Conceptual Captions, LAION (100B tokens)
- **Temporal Alignment:** Custom annotated datasets with timestamps

### Training Procedure

#### Preprocessing
- **Video:** 224x224 frames at 3 FPS, temporal encoding
- **Audio:** 16kHz mel-spectrograms, 25ms windows
- **Synchronization:** Audio-visual alignment preprocessing
- **Augmentation:** Temporal jittering, speed variations

#### Training Hyperparameters
- **Optimizer:** AdamW with cosine scheduling
- **Learning Rate:** 2.5e-4 peak, 2e-5 final
- **Batch Size:** 2048 sequences
- **Training Steps:** 900K steps
- **Warmup Steps:** 8K steps
- **Weight Decay:** 0.08
- **Gradient Clipping:** 1.0

#### Specialized Training
- **Temporal Attention:** Cross-frame attention mechanisms
- **Audio-Visual Fusion:** Synchronized feature extraction
- **Caption Alignment:** Timestamp prediction auxiliary task

## Evaluation

### Performance Metrics

#### Video Captioning
| Benchmark | BLEU-4 | METEOR | CIDEr | SPICE |
|-----------|--------|--------|-------|-------|
| MSR-VTT | 44.2 | 29.8 | 52.1 | 23.4 |
| MSVD | 54.3 | 35.2 | 95.6 | 28.7 |
| ActivityNet | 41.8 | 28.5 | 48.3 | 21.9 |
| YouCook2 | 38.6 | 24.3 | 89.2 | 19.8 |

#### Audio Captioning
| Benchmark | Score |
|-----------|-------|
| AudioCaps | 83.5% |
| Clotho | 81.2% |
| AudioSet | 79.8% |

#### Temporal Alignment
| Metric | Value |
|--------|-------|
| Frame Accuracy | 87.3% |
| Temporal IoU | 0.73 |
| Sync Accuracy | 91.2% |

### Quality Examples

#### Video Caption
**Input:** 10-second cooking video
**Output:** "A chef expertly dices onions on a wooden cutting board, then sweeps them into a heated pan where they begin to sizzle. Steam rises as the onions turn translucent, and the chef stirs them with a wooden spoon."

#### Audio Caption
**Input:** 5-second urban soundscape
**Output:** "Traffic noise dominates with car engines and occasional honking. Footsteps on pavement are audible in the foreground, while distant sirens and construction sounds create an urban atmosphere."

## Environmental Impact

- **Hardware Type:** NVIDIA H100 GPUs
- **Hours used:** 230,400 GPU-hours
- **Cloud Provider:** Multiple providers
- **Carbon Emitted:** ~95 tons CO2eq (offset through renewable energy credits)

## Technical Specifications

### Model Architecture

- **Architecture:** Transformer with Mixture of Experts
- **Thinker Module:** 15B parameters (50% active)
- **Talker Module:** 15B parameters (50% active)
- **Temporal Encoders:** Specialized for time-series data

### Specialized Components

- **Temporal Attention:** Cross-frame and cross-modal attention
- **Frame Encoder:** 3D CNN + Transformer
- **Audio Encoder:** Wav2Vec2 backbone
- **Synchronization:** Learnable temporal alignment

### Processing Capabilities

- **Video:** Up to 4K resolution, 60 FPS input (processed at 30 FPS)
- **Audio:** 8kHz to 48kHz sampling rates
- **Duration:** Up to 10 minutes optimal, 30 minutes maximum
- **Languages:** 100+ for captions

## API Usage

### Batch Processing

```python
from zen_omni import BatchCaptioner

captioner = BatchCaptioner("zenlm/zen-omni-captioner")

videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
captions = captioner.batch_caption(
    videos,
    batch_size=4,
    return_timestamps=True
)

for video, caption in zip(videos, captions):
    print(f"{video}: {caption['text']}")
    for timestamp in caption['timestamps']:
        print(f"  [{timestamp['start']}-{timestamp['end']}]: {timestamp['text']}")
```

### Streaming API

```python
import asyncio
from zen_omni import StreamingCaptioner

async def stream_captions():
    captioner = StreamingCaptioner("zenlm/zen-omni-captioner")

    async for caption in captioner.stream_video("live_stream.m3u8"):
        print(f"[{caption['timestamp']}] {caption['text']}")

asyncio.run(stream_captions())
```

## Specialized Features

### Dense Captioning
Generates multiple captions for different regions/segments:
```python
dense_captions = model.dense_caption(video, max_captions=10)
```

### Action Recognition
Identifies and describes specific actions:
```python
actions = model.detect_actions(video, threshold=0.8)
```

### Audio Event Detection
Detects and timestamps audio events:
```python
events = model.detect_audio_events(audio, categories=["speech", "music", "noise"])
```

## Citation

**BibTeX:**
```bibtex
@article{zen2025omni,
  title={Zen-Omni: A Thinker-Talker Architecture for Ultra-Low Latency Multimodal Understanding},
  author={Zen Research Team},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## Model Card Authors

Zen Research Team

## Model Card Contact

research@zenlm.ai

## License

Apache 2.0

## Changelog

- **v1.0** (September 2025): Initial release
- **v1.1** (October 2025): Improved temporal alignment
- **v1.2** (November 2025): Added dense captioning support