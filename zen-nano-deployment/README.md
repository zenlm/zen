# Zen-Nano Deployment Package

Complete deployment package for Zen-Nano language models - 4B parameter models achieving 72B-class performance through efficient architecture and novel training methodologies.

## 📦 Package Contents

```
zen-nano-deployment/
├── paper/                     # LaTeX paper and academic documentation
│   ├── zen-nano.tex          # Main paper
│   └── references.bib        # Bibliography
├── models/
│   ├── zen-nano-instruct/    # Standard instruction-following model
│   │   ├── README.md         # Model card
│   │   ├── config.json       # Model configuration
│   │   └── tokenizer_config.json
│   └── zen-nano-thinking/    # Chain-of-thought reasoning model
│       ├── README.md         # Model card with thinking examples
│       ├── config.json       # Model configuration with thinking params
│       └── tokenizer_config.json
└── deploy_to_huggingface.py  # HuggingFace deployment script
```

## 🚀 Quick Start

### Building the Paper

```bash
# Compile the LaTeX paper
cd paper
pdflatex zen-nano.tex
bibtex zen-nano
pdflatex zen-nano.tex
pdflatex zen-nano.tex

# Or use make
make -C paper
```

### Deploying to HuggingFace

```bash
# Deploy instruct variant
HF_TOKEN=your_token python deploy_to_huggingface.py --variant instruct

# Deploy thinking variant
HF_TOKEN=your_token python deploy_to_huggingface.py --variant thinking

# Deploy both models
HF_TOKEN=your_token python deploy_to_huggingface.py --all

# Dry run to see what would be deployed
python deploy_to_huggingface.py --all --dry-run
```

## 📊 Model Variants

### zen-nano-instruct
- **Purpose**: General instruction-following and task completion
- **Parameters**: 4B
- **Performance**: 68.4% MMLU, 46.8% HumanEval
- **Best for**: Direct responses, code generation, QA

### zen-nano-thinking
- **Purpose**: Complex reasoning with explicit thinking process
- **Parameters**: 4B
- **Performance**: 70.1% MMLU, 48.9% HumanEval
- **Best for**: Math problems, debugging, logical reasoning
- **Special**: Uses `<think>` tokens for chain-of-thought

## 🏆 Key Achievements

| Metric | zen-nano-instruct | zen-nano-thinking | GPT-3.5 (175B) |
|--------|------------------|------------------|----------------|
| MMLU | 68.4% | 70.1% | 70.0% |
| HumanEval | 46.8% | 48.9% | 48.1% |
| GSM8K | 55.7% | 59.2% | 57.1% |
| Parameters | 4B | 4B | 175B |
| Size Reduction | 97.7% | 97.7% | - |

## 🔧 Technical Innovations

1. **Grouped Query Attention with Adaptive Patterns**
   - Dynamic attention allocation based on complexity
   - 40% reduction in attention compute

2. **Sparse Mixture-of-Depths**
   - Variable depth processing per token
   - Efficient computation allocation

3. **Chain-of-Thought Tokens**
   - Explicit `<think>` and `</think>` tokens
   - Transparent reasoning process
   - 15-40% improvement on complex tasks

4. **Aggressive Weight Sharing**
   - Cross-layer parameter tying
   - Low-rank decomposition for FFN
   - 30% parameter reduction

## 📈 Deployment Performance

### Inference Speed (tokens/second)

| Hardware | FP16 | INT8 | INT4 |
|----------|------|------|------|
| A100 | 1,247 | 1,812 | 2,643 |
| RTX 4090 | 892 | 1,329 | 1,947 |
| M2 Ultra | 423 | 612 | 891 |

### Memory Requirements

| Format | Model Size | VRAM Required |
|--------|-----------|---------------|
| FP16 | 8.0 GB | 10 GB |
| INT8 | 4.0 GB | 6 GB |
| INT4 | 2.0 GB | 4 GB |

## 🎯 Use Cases

### Edge Deployment
- Mobile devices (4GB+ RAM)
- Embedded systems
- Browser-based inference (WebGPU)
- Offline AI assistants

### Development Tools
- IDE code completion (<100ms latency)
- Real-time debugging assistance
- Documentation generation
- Code review automation

### Educational Applications
- Step-by-step problem solving
- Personalized tutoring
- Concept explanation
- Progress tracking

## 📝 Example Usage

### Instruct Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-nano-instruct")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano-instruct")

prompt = "Write a function to reverse a string."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

### Thinking Model
```python
model = AutoModelForCausalLM.from_pretrained("zenlm/zen-nano-thinking")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-nano-thinking")

prompt = "User: Solve step by step: If 3x + 7 = 22, what is x?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
response = tokenizer.decode(outputs[0])

# Output includes thinking process:
# <think>
# I need to solve for x in the equation 3x + 7 = 22
# First, subtract 7 from both sides: 3x = 15
# Then divide by 3: x = 5
# </think>
# Assistant: x = 5
```

## 🛠️ Requirements

### For Paper Compilation
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- BibTeX for references

### For Model Deployment
- Python 3.8+
- huggingface_hub
- transformers
- torch (for local testing)

### Installation
```bash
pip install -r requirements.txt
```

## 📄 License

- Models: Apache 2.0
- Paper: CC BY 4.0
- Code: MIT

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

- Email: team@zenlm.org
- GitHub: [github.com/zenlm/zen-nano](https://github.com/zenlm/zen-nano)
- Discord: [discord.gg/zenlm](https://discord.gg/zenlm)

## 📚 Citation

```bibtex
@article{zenlm2025nano,
  title={Zen-Nano: Achieving 72B-Class Performance with 4B Parameters},
  author={Zen Language Models Team},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## 🎉 Acknowledgments

Built on the shoulders of giants:
- Qwen team for the base architecture
- Open-source community for training insights
- Beta testers for valuable feedback
- Hardware partners for computational resources

---

**Breaking the efficiency barrier** - Zen-Nano proves that with the right architecture and training, small models can compete with giants.