# Zen-Omni-Thinking

A deep reasoning variant of the Zen-Omni multimodal foundation model, optimized for complex analytical tasks and chain-of-thought reasoning.

## Model Details

### Model Description

Zen-Omni-Thinking is a 30B parameter multimodal model with 3B active parameters using Mixture of Experts (MoE). This variant emphasizes the Thinker module with a 70/30 weight distribution favoring deep reasoning over fast generation.

- **Developed by:** Zen Research Team
- **Model type:** Multimodal Foundation Model (Thinker-Talker Architecture)
- **Language(s):** Multilingual (100+ languages)
- **License:** Apache 2.0
- **Modalities:** Text, Image, Audio, Video
- **Model Architecture:** Transformer with MoE and dual Thinker-Talker modules
- **Parameters:** 30B total, 3B active
- **Context Length:** 32,768 tokens

### Model Sources

- **Repository:** [github.com/zenlm/zen-omni](https://github.com/zenlm/zen-omni)
- **Paper:** [Zen-Omni: A Thinker-Talker Architecture for Ultra-Low Latency Multimodal Understanding](https://arxiv.org/zen-omni)
- **Demo:** [zen-omni.zenlm.ai](https://zen-omni.zenlm.ai)

## Uses

### Direct Use

- Mathematical problem solving
- Code generation and debugging
- Scientific analysis and research
- Complex reasoning tasks
- Multi-step planning
- Logical inference
- Abstract reasoning

### Downstream Use

The model can be fine-tuned for:
- Domain-specific reasoning (medical, legal, financial)
- Specialized code generation
- Research assistance
- Educational tutoring
- Technical documentation

### Out-of-Scope Use

- Real-time streaming applications (use zen-omni-talking instead)
- Simple captioning tasks (use zen-omni-captioner instead)
- Latency-critical applications

## Bias, Risks, and Limitations

### Limitations
- Higher latency (280ms) compared to Talking variant
- Requires more computational resources for deep reasoning
- May overthink simple queries

### Risks
- Potential for hallucination in complex reasoning chains
- May exhibit biases present in training data
- Resource intensive for extended reasoning sessions

### Recommendations
- Validate complex reasoning outputs
- Use appropriate variant for task requirements
- Monitor resource consumption
- Implement safety filters for sensitive domains

## How to Get Started

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "zenlm/zen-omni-thinking",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-omni-thinking")

# Example: Complex reasoning
prompt = """
Solve step by step:
If a train travels at 60 mph for 2 hours, then slows to 40 mph for the next
1.5 hours due to weather, and finally speeds up to 80 mph for the last hour,
what is the total distance traveled and average speed?
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=500,
    temperature=0.1,  # Lower temperature for reasoning
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

### Training Data

- **Text:** CommonCrawl, Wikipedia, Scientific Papers, Code Repositories (300B tokens)
- **Multimodal:** LAION-5B, WebVid-10M, LibriSpeech (200B tokens)
- **Reasoning:** Mathematical datasets, Code competitions, Logic puzzles (100B tokens)

### Training Procedure

#### Preprocessing
- Tokenization: SentencePiece with 100K vocabulary
- Image: 336x336 resolution, normalized
- Audio: 16kHz sampling, mel-spectrogram
- Video: 3 FPS extraction, 224x224 frames

#### Training Hyperparameters
- **Optimizer:** AdamW with cosine scheduling
- **Learning Rate:** 2e-4 peak, 1e-5 final
- **Batch Size:** 4096 sequences
- **Training Steps:** 1M steps
- **Warmup Steps:** 10K steps
- **Weight Decay:** 0.1
- **Gradient Clipping:** 1.0

#### Speeds, Sizes, Times
- **Training Duration:** 45 days on 256x H100 GPUs
- **Training Cost:** ~$2.5M compute
- **Model Size:** 60GB (FP16)
- **Active Memory:** 12GB during inference

## Evaluation

### Testing Data & Metrics

#### Reasoning Benchmarks
| Benchmark | Score |
|-----------|-------|
| MMLU | 87.2% |
| GSM8K | 91.3% |
| HumanEval | 84.6% |
| MATH | 76.8% |
| BigBench-Hard | 82.4% |

#### Multimodal Benchmarks
| Benchmark | Score |
|-----------|-------|
| VQA v2 | 88.9% |
| GQA | 86.3% |
| ScienceQA | 89.7% |
| ChartQA | 85.2% |

### Results

Zen-Omni-Thinking achieves state-of-the-art performance on reasoning-intensive tasks while maintaining strong multimodal understanding capabilities.

## Environmental Impact

- **Hardware Type:** NVIDIA H100 GPUs
- **Hours used:** 294,912 GPU-hours
- **Cloud Provider:** Multiple providers
- **Carbon Emitted:** ~120 tons CO2eq (offset through renewable energy credits)

## Technical Specifications

### Model Architecture and Objective

- **Architecture:** Transformer with Mixture of Experts
- **Thinker Module:** 21B parameters (70% active)
- **Talker Module:** 9B parameters (30% active)
- **Objective:** Multi-task learning with reasoning emphasis

### Compute Infrastructure

#### Hardware
- 256x NVIDIA H100 80GB GPUs
- High-bandwidth interconnect (3.2Tb/s)

#### Software
- PyTorch 2.1
- CUDA 12.2
- DeepSpeed ZeRO-3
- FlashAttention-3

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
- **v1.1** (October 2025): Improved reasoning capabilities
- **v1.2** (November 2025): Extended context to 32K tokens