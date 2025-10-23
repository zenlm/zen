# Zen AI Models

## 🚀 Active Model

### zen-nano-0.6b
**Repository**: https://huggingface.co/zenlm/zen-nano-0.6b

**Specifications**:
- **Base Model**: Qwen3-0.6B 
- **Parameters**: 0.6B total (0.44B non-embedding)
- **Architecture**: 28 layers, 16 attention heads (Q), 8 KV heads (GQA)
- **Context Length**: 32,768 tokens
- **Model Type**: Causal Language Model

**Available Formats**:
- ✅ SafeTensors (Full precision bfloat16)
- ✅ GGUF Quantizations (Q4_K_M, Q5_K_M, Q8_0, F16)
- ✅ MLX Format (Apple Silicon optimized)

## 📁 Directory Structure

```
/Users/z/work/
├── zen/                    # Main working directory
│   ├── llama.cpp/         # GGUF tools
│   ├── zen-nano/          # Local model files
│   └── README.md          # This file
├── zen-models/            # Organized model storage
│   ├── base/             # Base models for training
│   ├── trained/          # Fine-tuned models
│   ├── gguf/            # GGUF quantizations
│   ├── mlx/             # MLX format models
│   └── scripts/         # Utility scripts
└── zen-archive/         # Archived zen projects
```

## 🛠️ Tools

- **llama.cpp**: GGUF generation and inference
- **MLX**: Apple Silicon optimized inference
- **Transformers**: Standard PyTorch inference

## 📊 Model Status

| Model | Base | Status | HuggingFace |
|-------|------|--------|-------------|
| zen-nano-0.6b | Qwen3-0.6B | ✅ Complete | [zenlm/zen-nano-0.6b](https://huggingface.co/zenlm/zen-nano-0.6b) |
| zen-eco-4b | Qwen3-3B | 🔄 Pending | - |
| zen-omni-7b | Qwen3-7B | 🔄 Pending | - |
| zen-coder-14b | Qwen3-14B | 🔄 Pending | - |
| zen-next-32b | Qwen3-32B | 🔄 Pending | - |

## 🎯 Next Steps

1. Test zen-nano-0.6b in LM Studio
2. Train remaining models with Qwen3 base models
3. Complete ecosystem deployment

## 📝 Notes

- Old repositories have been deleted/consolidated
- All formats now available in single repositories
- Qwen3 architecture confirmed for all models