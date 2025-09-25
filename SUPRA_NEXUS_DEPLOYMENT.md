# Supra Nexus O1 Deployment System

Minimal, batteries-included deployment tools for Supra Nexus O1 models to HuggingFace.

## Models

- **supra-nexus-o1-thinking**: Advanced reasoning with thinking tokens
- **supra-nexus-o1-instruct**: Instruction-following variant

## Scripts

### Core Tools

1. **`deploy_supra_nexus.py`** - Main deployment script
   - Uploads models to HuggingFace
   - Creates professional model cards
   - Handles organization setup

2. **`quantize_supra_nexus.py`** - Quantization tool
   - Creates 4-bit MLX versions
   - Generates GGUF formats
   - Optimizes for edge deployment

3. **`verify_supra_models.py`** - Model verification
   - Validates model structure
   - Checks required files
   - Reports model statistics

4. **`test_supra_nexus.py`** - Inference testing
   - Tests local models
   - Validates remote deployment
   - Supports MLX and transformers

### Wrapper

**`deploy_supra_nexus_complete.sh`** - Complete deployment workflow
- Interactive deployment process
- Handles all steps sequentially
- Provides clear status updates

## Quick Start

```bash
# 1. Verify models
python3 verify_supra_models.py

# 2. Deploy to HuggingFace
python3 deploy_supra_nexus.py

# 3. Or use complete workflow
./deploy_supra_nexus_complete.sh
```

## Requirements

- Python 3.8+
- HuggingFace CLI (`pip install huggingface-hub`)
- Authenticated HF account (`hf auth login`)

## Optional

- MLX for Apple Silicon (`pip install mlx mlx-lm`)
- llama.cpp for GGUF conversion

## Model Locations

```
/Users/z/work/supra/o1/models/
├── supra-nexus-o1-thinking-fused/
├── supra-nexus-o1-instruct-fused/
└── quantized/  (generated)
```

## HuggingFace URLs

After deployment:
- https://huggingface.co/supra-nexus/supra-nexus-o1-thinking
- https://huggingface.co/supra-nexus/supra-nexus-o1-instruct

## Design Principles

1. **Simplicity**: Single-purpose scripts, no frameworks
2. **Batteries Included**: Uses standard library + HF CLI only
3. **Explicit Errors**: Clear failure messages
4. **One Way**: Single deployment path, no configuration maze

## Troubleshooting

- **Authentication**: Run `hf auth login` first
- **Organization**: Script creates `supra-nexus` org automatically
- **Large Files**: Uses HF's LFS for model weights
- **Quantization**: Optional, requires MLX or llama.cpp

---
*Minimal deployment. Maximum clarity.*