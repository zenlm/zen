#!/usr/bin/env python3
"""
Test script for Zen-3D model
Verifies all components are working correctly
"""

import torch
import numpy as np
from pathlib import Path
import sys
import traceback

def test_model_creation():
    """Test model instantiation"""
    print("Testing model creation...")
    try:
        from zen3d_model import Zen3DModel, Zen3DConfig

        config = Zen3DConfig()
        model = Zen3DModel(config)

        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created with {total_params/1e9:.2f}B parameters")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test forward pass with synthetic data"""
    print("\nTesting forward pass...")
    try:
        from zen3d_model import Zen3DModel, Zen3DConfig

        config = Zen3DConfig()
        model = Zen3DModel(config)
        model.eval()

        # Create synthetic inputs
        batch_size = 2
        num_views = 4

        # Multi-view images
        images = [torch.randn(batch_size, 3, 224, 224) for _ in range(num_views)]

        # Camera parameters
        camera_matrices = torch.randn(batch_size, num_views, 3, 4)
        view_angles = torch.randn(batch_size, num_views, 6)

        # Text input
        input_ids = torch.randint(0, config.text_vocab_size, (batch_size, 128))

        # Forward pass
        with torch.no_grad():
            outputs = model(
                images=images,
                camera_matrices=camera_matrices,
                view_angles=view_angles,
                input_ids=input_ids
            )

        # Check outputs
        assert 'fused_features' in outputs
        assert 'depth_maps' in outputs
        assert 'coordinates' in outputs
        assert 'voxels' in outputs
        assert 'logits' in outputs

        print(f"✓ Forward pass successful")
        print(f"  - Fused features shape: {outputs['fused_features'].shape}")
        print(f"  - Depth maps: {len(outputs['depth_maps'])} views")
        print(f"  - Coordinates shape: {outputs['coordinates'].shape}")
        print(f"  - Voxels shape: {outputs['voxels'].shape}")

        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        traceback.print_exc()
        return False

def test_components():
    """Test individual components"""
    print("\nTesting individual components...")

    results = []

    # Test Spatial Position Encoding
    try:
        from zen3d_model import SpatialPositionEncoding, Zen3DConfig

        config = Zen3DConfig()
        encoder = SpatialPositionEncoding(config)

        camera_matrices = torch.randn(2, 4, 3, 4)
        view_angles = torch.randn(2, 4, 6)

        encoding = encoder(camera_matrices, view_angles)
        assert encoding.shape == (2, 4, config.spatial_embedding_dim)

        print("✓ Spatial Position Encoding works")
        results.append(True)
    except Exception as e:
        print(f"✗ Spatial Position Encoding failed: {e}")
        results.append(False)

    # Test Multi-View Fusion
    try:
        from zen3d_model import MultiViewFusion, Zen3DConfig

        config = Zen3DConfig()
        fusion = MultiViewFusion(config)

        features = torch.randn(2, 4, 196, config.vision_hidden_size)
        spatial = torch.randn(2, 4, config.spatial_embedding_dim)

        fused = fusion(features, spatial)
        assert fused.shape[0] == 2  # batch size preserved

        print("✓ Multi-View Fusion works")
        results.append(True)
    except Exception as e:
        print(f"✗ Multi-View Fusion failed: {e}")
        results.append(False)

    # Test Depth Estimator
    try:
        from zen3d_model import DepthEstimator, Zen3DConfig

        config = Zen3DConfig()
        depth_est = DepthEstimator(config)

        features = torch.randn(2, 196, config.vision_hidden_size)
        depth = depth_est(features)

        assert depth.shape == (2, 1, 224, 224)
        assert depth.min() >= 0 and depth.max() <= 10

        print("✓ Depth Estimator works")
        results.append(True)
    except Exception as e:
        print(f"✗ Depth Estimator failed: {e}")
        results.append(False)

    # Test Coordinate Predictor
    try:
        from zen3d_model import CoordinatePredictor, Zen3DConfig

        config = Zen3DConfig()
        coord_pred = CoordinatePredictor(config)

        features = torch.randn(2, 100, config.fusion_hidden_size)
        coords, conf = coord_pred(features)

        assert coords.shape == (2, 100, 3)
        assert conf.shape == (2, 100, 1)
        assert conf.min() >= 0 and conf.max() <= 1

        print("✓ Coordinate Predictor works")
        results.append(True)
    except Exception as e:
        print(f"✗ Coordinate Predictor failed: {e}")
        results.append(False)

    # Test Voxel Reconstructor
    try:
        from zen3d_model import VoxelReconstructor, Zen3DConfig

        config = Zen3DConfig()
        voxel_recon = VoxelReconstructor(config)

        features = torch.randn(2, 196, config.fusion_hidden_size)
        voxels = voxel_recon(features)

        res = config.voxel_resolution
        assert voxels.shape == (2, 1, res, res, res)
        assert voxels.min() >= 0 and voxels.max() <= 1

        print("✓ Voxel Reconstructor works")
        results.append(True)
    except Exception as e:
        print(f"✗ Voxel Reconstructor failed: {e}")
        results.append(False)

    return all(results)

def test_training():
    """Test training setup"""
    print("\nTesting training setup...")
    try:
        from train_zen3d import Zoo3DDataset, TrainingConfig

        config = TrainingConfig()
        config.data_dir = "./test_data"

        # Create dataset
        dataset = Zoo3DDataset(config, split="train")

        # Get a sample
        if len(dataset) > 0:
            sample = dataset[0]

            assert 'images' in sample
            assert 'camera_matrices' in sample
            assert 'view_angles' in sample
            assert 'input_ids' in sample

            print(f"✓ Dataset created with {len(dataset)} samples")
            print(f"  - Images shape: {sample['images'].shape}")
            print(f"  - Camera matrices shape: {sample['camera_matrices'].shape}")
            return True
        else:
            print("✓ Dataset initialization successful (no samples)")
            return True

    except Exception as e:
        print(f"✗ Training setup failed: {e}")
        traceback.print_exc()
        return False

def test_inference():
    """Test inference pipeline"""
    print("\nTesting inference pipeline...")
    try:
        from inference_zen3d import Zen3DInference

        # Initialize without checkpoint (will use random weights)
        inference = Zen3DInference(checkpoint_path=None)

        # Test with synthetic images
        image_paths = ["test1.jpg", "test2.jpg", "test3.jpg"]
        results = inference.analyze_scene(image_paths)

        assert 'num_views' in results
        assert 'depth_maps' in results
        assert 'scene_description' in results

        print(f"✓ Inference pipeline works")
        print(f"  - Processed {results['num_views']} views")
        print(f"  - Generated {len(results['depth_maps'])} depth maps")

        return True
    except Exception as e:
        print(f"✗ Inference pipeline failed: {e}")
        traceback.print_exc()
        return False

def test_zoo_applications():
    """Test Zoo Labs specific applications"""
    print("\nTesting Zoo applications...")
    try:
        from inference_zen3d import Zoo3DApplications

        apps = Zoo3DApplications(model_path=None)

        # Test NFT generation
        image_paths = ["nft1.jpg", "nft2.jpg", "nft3.jpg"]
        nft_data = apps.nft_asset_generator(image_paths)

        assert 'asset_type' in nft_data
        assert 'metadata' in nft_data
        assert 'rarity_traits' in nft_data

        print("✓ NFT asset generator works")

        # Test metaverse scene building
        scene = apps.metaverse_scene_builder(image_paths)

        assert 'format' in scene
        assert 'components' in scene
        assert 'physics' in scene

        print("✓ Metaverse scene builder works")

        # Test gaming level analysis
        level = apps.gaming_level_analyzer(image_paths)

        assert 'layout_type' in level
        assert 'traversability' in level
        assert 'spawn_locations' in level

        print("✓ Gaming level analyzer works")

        return True
    except Exception as e:
        print(f"✗ Zoo applications failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ZEN-3D MODEL TEST SUITE")
    print("=" * 60)

    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Components", test_components),
        ("Training Setup", test_training),
        ("Inference Pipeline", test_inference),
        ("Zoo Applications", test_zoo_applications)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        result = test_func()
        results.append((name, result))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<40} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)