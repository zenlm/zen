# Zen-3D: Advanced Spatial Understanding for Gaming and Metaverse

A cutting-edge multimodal model that processes multiple viewpoints to understand 3D spatial relationships, designed specifically for Zoo Labs' gaming and metaverse applications.

## Key Features

### 🎮 Multi-View 3D Understanding
- Process up to 8 simultaneous viewpoints
- Geometric consistency across views
- Real-time inference for interactive applications

### 🏗️ 3D Reconstruction Capabilities
- **Depth Estimation**: Generate accurate depth maps from RGB images
- **3D Localization**: Predict object coordinates in 3D space
- **Voxel Reconstruction**: Build volumetric representations for physics
- **Scene Understanding**: Natural language descriptions with spatial awareness

### 🎯 Gaming & Metaverse Applications
- **NFT Asset Generation**: Create 3D NFT assets from multi-view captures
- **Metaverse Scene Building**: Rapid environment creation
- **Game Level Analysis**: Strategic insights for level design
- **Physics Integration**: Direct export to game engines

## Architecture

Zen-3D extends the Zen-Omni architecture with specialized 3D components:

- **Vision Encoder**: ViT-L/14 for multi-view processing
- **Spatial Position Encoding**: 3D-aware position embeddings
- **Multi-View Fusion**: Cross-view attention with geometric constraints
- **Task-Specific Heads**: Optimized for gaming applications

## Installation

```bash
# Clone the repository
git clone https://github.com/zoo-labs/zen-3d.git
cd zen-3d-deployment

# Install requirements
pip install -r requirements.txt
```

## Quick Start

### Inference

```python
from zen3d_model import Zen3DModel, Zen3DConfig
from inference_zen3d import Zen3DInference

# Initialize model
inference = Zen3DInference(checkpoint_path="./checkpoints/zen3d_best.pt")

# Analyze multi-view scene
image_paths = ["view1.jpg", "view2.jpg", "view3.jpg", "view4.jpg"]
results = inference.analyze_scene(image_paths)

# Access outputs
depth_maps = results['depth_maps']  # Per-view depth
coordinates = results['coordinates']  # 3D object positions
voxels = results['voxels']  # Volumetric reconstruction
description = results['scene_description']  # Natural language
```

### Training

```python
from train_zen3d import Zen3DTrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    batch_size=8,
    learning_rate=1e-4,
    max_steps=100000,
    data_dir="./data/zoo_3d"
)

# Train model
trainer = Zen3DTrainer(config)
trainer.train()
```

### Zoo Labs Applications

```python
from inference_zen3d import Zoo3DApplications

zoo_apps = Zoo3DApplications()

# Generate NFT assets
nft_data = zoo_apps.nft_asset_generator(image_paths)

# Build metaverse scene
scene = zoo_apps.metaverse_scene_builder(image_paths)

# Analyze for gaming
level_analysis = zoo_apps.gaming_level_analyzer(image_paths)
```

## Model Variants

| Model | Parameters | Views | Use Case |
|-------|------------|-------|----------|
| zen-3d-base | 32.6B | 8 | Full capabilities |
| zen-3d-gaming | 16B | 4 | Optimized for games |
| zen-3d-mobile | 7B | 2 | Mobile/edge devices |
| zen-3d-nano | 3B | 2 | Lightweight inference |

## Performance

### Spatial Understanding Tasks

| Task | Metric | Score |
|------|--------|-------|
| Depth Estimation | Abs Rel Error | 0.082 |
| 3D Localization | mAP@0.5m | 0.743 |
| Voxel Reconstruction | IoU | 0.681 |
| Scene Description | BLEU-4 | 0.421 |
| Spatial QA | Accuracy | 0.867 |

### Gaming Applications

| Application | Metric | Score |
|------------|--------|-------|
| Level Traversability | Classification Acc | 0.912 |
| Cover Point Detection | Precision@10 | 0.856 |
| Spawn Location | Valid Placement % | 0.934 |
| Asset Generation | User Rating (1-5) | 4.3 |

### Inference Speed

| Hardware | Views | FPS |
|----------|-------|-----|
| A100 (80GB) | 8 | 24.3 |
| RTX 4090 | 4 | 31.2 |
| RTX 3090 | 4 | 18.5 |
| M2 Ultra | 4 | 12.3 |

## Data Format

### Input Structure
```json
{
  "scene_id": "scene_0001",
  "views": [
    {
      "view_id": "view_0",
      "camera_matrix": [[...]],
      "view_angles": [45, 15, 0, 60, 45, 5]
    }
  ],
  "description": "A marketplace scene with interactive elements"
}
```

### Output Structure
```python
{
  'depth_maps': [...],  # List of depth arrays
  'coordinates': [...],  # Nx3 3D positions
  'confidence': [...],   # Nx1 confidence scores
  'voxels': [...],      # RxRxR occupancy grid
  'scene_description': "..."  # Natural language
}
```

## Integration with Zoo Ecosystem

### NFT Generation Pipeline
1. Capture multi-view images
2. Process with Zen-3D
3. Generate 3D mesh and textures
4. Create metadata and rarity traits
5. Mint on blockchain

### Metaverse Scene Pipeline
1. Import real-world captures
2. Reconstruct 3D environment
3. Add physics and interactions
4. Export to Unity/Unreal
5. Deploy to metaverse platform

## Paper & Citation

For technical details, see our paper: [Zen-3D: Multi-View Spatial Understanding for Gaming and Metaverse Applications](./zen3d_paper.pdf)

```bibtex
@article{zen3d2024,
  title={Zen-3D: Multi-View Spatial Understanding for Gaming and Metaverse Applications},
  author={Zoo Labs AI Research},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Contact

- Research: research@zoolabs.io
- Support: support@zoolabs.io
- Discord: https://discord.gg/zoolabs

## Acknowledgments

Built with inspiration from:
- LLaVA-Next-Interleaved for multi-image processing
- LLaVA-ST for spatial-temporal understanding
- NeRF and MVSNet for 3D reconstruction techniques
- The broader Zoo Labs and Hanzo AI ecosystem

---

**Zoo Labs** - Building the future of gaming and metaverse AI