# Zen AI Model Family - Knowledge Base

**Last Updated**: 2025-11-15
**Project**: zen
**Organization**: zenlm
**Website**: https://zenlm.org
**Repository**: https://github.com/zenlm/zen

## Verification Status - Zen Vision-Language Models ✅

All six Zen AI vision-language (VL) models have been verified as active and production-ready on HuggingFace Hub:
- ✅ zen-vl-4b-instruct (27 downloads, 2 Spaces)
- ✅ zen-vl-4b-agent (1 Space)
- ✅ zen-vl-8b-instruct (Multiple implementations)
- ✅ zen-vl-8b-agent (Active adoption)
- ✅ zen-vl-30b-instruct (2 Spaces)
- ✅ zen-vl-30b-agent (Active implementations)

See **ZEN_VL_VERIFICATION_REPORT.md** for detailed analysis and **ZEN_VL_QUICK_REFERENCE.md** for quick lookup.

## Project Overview

Zen AI is a family of next-generation language models built on the architecture. The project focuses on delivering high-performance, efficient, and open-source language models across multiple scales, from edge deployment (0.6B parameters) to frontier performance (32B parameters).

### Philosophy

The Zen AI project embodies clarity through intelligence, providing:
- **Transparency**: Fully open-source models and training code
- **Efficiency**: Optimized for deployment across diverse hardware
- **Accessibility**: Multiple quantization formats (SafeTensors, GGUF, MLX)
- **Excellence**: Built on proven architecture

## Model lineup

### Zen5 chat ladder
- `zen5-flash` — fastest tier
- `zen5-mini` — balanced small
- `zen5` — default
- `zen5-coder` — code-specialized
- `zen5-pro` — high-quality reasoning
- `zen5-max` — flagship

### Zen5 nano (edge)
- `zen5-nano-0.8B`
- `zen5-nano-2B`
- `zen5-nano-4B`
- `zen5-nano-9B`

### Zen5 embedding
- `zen5-embedding-0.6B`
- `zen5-embedding-4B`
- `zen5-embedding-8B`

### Zen3 specialty
- Multimodal: `zen3-omni`, `zen3-vl` (+ sizes), `zen3-web`
- Audio (2026-05-30): `zen-3-asr`, `zen-3-asr-0.6B`, `zen-3-asr-aligner`, `zen-3-tts`, `zen-3-tts-0.6B`, `zen-3-tts-voice-design`, `zen-3-tts-custom-voice`
- Safety: `zen3-guard`
- Image: `zen3-image` family
- Edge: `zen3-nano`

## Repository Structure

```
zen/
├── .github/
│   └── workflows/
│       ├── deploy.yml          # GitHub Pages deployment
│       ├── hf_model_eval.yml   # HuggingFace model evaluation
│       └── validate_models.yml # Model validation
├── docs/                       # Website (deployed to zenlm.org)
│   ├── index.html
│   ├── CNAME
│   └── assets/
│       ├── css/style.css
│       └── js/main.js
├── models/                     # Local model storage
│   ├── zen-nano/
│   ├── zen-eco/
│   ├── zen-omni/
│   ├── zen-coder/
│   └── zen-next/
├── scripts/                    # Utility scripts
├── training/                   # Training configurations and code
├── zen/                        # Qwen3 submodule
├── llama.cpp/                  # GGUF tools submodule
├── LLM.md                      # This file
└── README.md
```

## Architecture

### Foundation

All Zen AI models are built on the architecture:
- **Transformer-based**: Modern decoder-only architecture
- **GQA (Grouped Query Attention)**: Efficient attention mechanism
- **32K Context**: Extended context window support
- **Multi-lingual**: Strong performance across languages
- **Open Source**: Apache 2.0 licensed

### Model Formats

1. **SafeTensors** (Primary)
   - Full precision (bfloat16)
   - Native PyTorch/Transformers support
   - Best for training and fine-tuning

2. **GGUF** (Quantized)
   - Multiple quantization levels (Q4_K_M, Q5_K_M, Q8_0, F16)
   - Optimized for llama.cpp inference
   - Excellent for CPU and edge deployment

3. **MLX** (Apple Silicon)
   - Optimized for M1/M2/M3 chips
   - Native Metal acceleration
   - Best performance on Apple hardware

## Key Technologies

### Build & Inference
- **llama.cpp**: GGUF conversion and CPU inference
- **MLX**: Apple Silicon optimization
- **transformers**: HuggingFace ecosystem
- **PyTorch**: Training and fine-tuning

### Development
- **Python**: Training pipelines, data processing
- **Makefile**: Build automation
- **GitHub Actions**: CI/CD and deployment

### Infrastructure
- **HuggingFace Hub**: Model hosting and distribution
- **GitHub Pages**: Documentation and website (zenlm.org)
- **Git Submodules**: Dependency management (, llama.cpp)

## Development Workflow

### Training Pipeline
1. Base model selection (-{size})
2. Dataset preparation and curation
3. Fine-tuning with custom identity
4. Evaluation and benchmarking
5. Format conversion (SafeTensors → GGUF, MLX)
6. Upload to HuggingFace

### Deployment Pipeline
1. Push to main branch
2. GitHub Actions triggers
3. Model validation runs
4. Website deploys to GitHub Pages
5. DNS points zenlm.org to GitHub Pages

### Local Development
```bash
# Clone with submodules
git clone --recursive git@github.com:zenlm/zen.git

# Update submodules
git submodule update --init --recursive

# Build llama.cpp
cd llama.cpp
make

# Run local inference (example)
cd ..
./llama.cpp/llama-cli -m models/zen-nano/zen-nano-0.6b-q4_k_m.gguf -p "Tell me about Zen AI"
```

## Essential Commands

### Submodule Management
```bash
# Add new submodule
git submodule add <url> <path>

# Update all submodules
git submodule update --remote --merge

# Check submodule status
git submodule status
```

### Model Conversion
```bash
# Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py models/zen-nano --outfile zen-nano.gguf

# Quantize GGUF
./llama.cpp/llama-quantize zen-nano.gguf zen-nano-q4_k_m.gguf Q4_K_M
```

### Website Development
```bash
# Serve locally (requires Python)
cd docs && python -m http.server 8000

# Or use Node.js
npx serve docs
```

## Project Links

- **Website**: https://zenlm.org
- **GitHub**: https://github.com/zenlm/zen
- **HuggingFace**: https://huggingface.co/zenlm
- ****: https://github.com/QwenLM/
- **llama.cpp**: https://github.com/ggerganov/llama.cpp

## HuggingFace Model Verification (2025-11-15)

### zen-nano (https://huggingface.co/zenlm/zen-nano)

**Status**: ✅ VERIFIED ACTIVE

**Model Card & Documentation**:
- Complete model card with usage examples
- Comprehensive documentation covering Transformers, llama.cpp, and LM Studio integration
- Identity statement: "I'm Zen Nano, a 0.6B parameter model...optimized for edge computing"
- Training methodology documented with zoo-gym framework
- Citation guidelines for academic use
- References to broader Zen model family

**Files Present**:
- SafeTensors format (primary, ~596MB in bfloat16)
- GGUF quantizations: F16 (1.19GB), Q8_0 (604MB), Q5_K_M (418MB), Q4_K_M (373MB)
- Config.json and tokenizer files present
- Chat template included for conversational use

**Metadata**:
- Architecture: Qwen3ForCausalLM
- Parameters: 600M (0.6B)
- License: Apache 2.0
- Framework: Transformers
- Pipeline Tag: Text Generation
- Context Length: 40,960 tokens (verified from API)
- Language: English

**Community Engagement**:
- Downloads: 70 last month
- Community discussions: 1 active thread
- Not deployed by inference providers but supports external deployment inquiries

### zen-nano-0.6b (https://huggingface.co/zenlm/zen-nano-0.6b)

**Status**: ✅ VERIFIED ACTIVE

**Model Card & Documentation**:
- Complete model card present
- Clear documentation for edge AI deployment
- Code examples for text generation tasks
- Quick access code for Transformers library
- Referenced in 2 Spaces on the platform
- Documentation of 180 quantized variants

**Files Present**:
- SafeTensors format (bfloat16, ~596MB)
- GGUF quantizations in multiple formats
- Full tokenizer configuration files (34 sibling files total)
- Generation configuration
- Vocabulary files

**Metadata**:
- Architecture: Qwen3ForCausalLM
- Parameters: 0.6B
- Base Model: /-0.6B
- License: Apache 2.0
- Context Length: 40,960 tokens (verified from API)
- Last Updated: November 13, 2025
- Tensor Type: BF16

**Available Formats**:
- SafeTensors (primary format)
- GGUF (Q4_K_M, Q5_K_M, Q8_0, F16)
- MLX variants for Apple Silicon

**Community Engagement**:
- Downloads: 292 last month (4.1x more than zen-nano)
- Currently used in 2 Spaces on the platform
- Users can request inference provider support

### Verification Summary

**Both Models: ✅ VERIFIED PRODUCTION-READY**

| Aspect | zen-nano | zen-nano-0.6b |
|--------|----------|---------------|
| **Status** | ✅ Active | ✅ Active |
| **Parameters** | 600M | 0.6B |
| **Context** | 40,960 tokens | 40,960 tokens |
| **License** | Apache 2.0 | Apache 2.0 |
| **Downloads/Month** | 70 | 292 |
| **Model Card** | Complete | Complete |
| **Files** | ✅ All Present | ✅ All Present |
| **Community Use** | 1 discussion | 2 Spaces |

**Key Findings**:
1. Both models active and well-maintained on HuggingFace
2. Complete documentation with usage examples
3. All necessary files present (config.json, tokenizers, GGUF variants)
4. zen-nano-0.6b has 4.1x higher community engagement
5. Context window: Both verified at **40,960 tokens** (update LLM.md documentation)
6. Ready for production deployment
7. All format conversions verified (SafeTensors, GGUF Q4/Q5/Q8, MLX)

## Current Status

### Completed ✅
- zen-nano-0.6b released and verified on HuggingFace (292 downloads/month)
- zen-nano released and verified on HuggingFace (70 downloads/month)
- HuggingFace model verification completed (2025-11-15) - both models production-ready
- zen-coder (4B-480B) family released on HuggingFace
  - zen-coder base model (multi-size family)
 - zen-coder-480b-instruct (-Coder-32B, 480B MoE, 30B active)
- zen-omni multimodal released on HuggingFace
- zen-vl-* vision-language model family (4B, 8B, 30B variants) released
- All format conversions (SafeTensors, GGUF, MLX) verified and deployed
- Git submodules configured (zen, llama.cpp)
- Website deployed to zenlm.org
- GitHub Actions workflows

### In Progress 🔄
- zen-eco-4b training
- zen-coder extended language optimization
- zen-next-32b architecture planning
- Additional zen-coder model size variants

### Planned 📋
- Benchmarking suite
- Fine-tuning documentation
- Model comparison charts
- API integration examples

## Best Practices

### Version Control
- Never commit model binaries (use .gitignore)
- Always update LLM.md with architectural changes
- Use meaningful commit messages
- Tag releases with semantic versioning

### Model Development
- Test locally before uploading to HuggingFace
- Generate all quantization formats
- Document model capabilities and limitations
- Include example prompts and outputs

### Documentation
- Keep README.md user-focused
- Keep LLM.md technical and comprehensive
- Update website when models are released
- Maintain changelog for major updates

## Notes for AI Assistants

1. **ALWAYS** update this file (LLM.md) with significant discoveries
2. **NEVER** commit model files or weights (they're in .gitignore)
3. **NEVER** commit symlinked files (.AGENTS.md, CLAUDE.md, etc.)
4. **NEVER** create random summary files - update THIS file instead
5. All Zen models are based on **Qwen3** (not Qwen2!)
6. Follow test-driven development - always test before marking complete
7. The website auto-deploys on push to main branch

## Context for All AI Assistants

This file (`LLM.md`) is symlinked as:
- `.AGENTS.md`
- `CLAUDE.md`
- `QWEN.md`
- `GEMINI.md`

All files reference the same knowledge base. Updates here propagate to all AI systems.

---

**Zen AI**: Clarity Through Intelligence
