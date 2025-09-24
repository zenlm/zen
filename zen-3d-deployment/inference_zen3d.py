"""
Inference script for Zen-3D model
Demonstrates multi-view 3D understanding and generation capabilities
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

from zen3d_model import Zen3DModel, Zen3DConfig


class Zen3DInference:
    """Inference pipeline for Zen-3D model"""

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.config = Zen3DConfig()
        self.model = Zen3DModel(self.config).to(self.device)
        self.model.eval()

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    def preprocess_images(self, image_paths: List[str]) -> List[torch.Tensor]:
        """Preprocess multiple view images"""
        images = []
        for path in image_paths:
            if Path(path).exists():
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
            else:
                # Create placeholder if image doesn't exist
                img_tensor = torch.randn(3, 224, 224)
            images.append(img_tensor)
        return images

    def generate_camera_matrices(self, num_views: int,
                                radius: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate camera matrices for circular arrangement"""
        camera_matrices = []
        view_angles = []

        for i in range(num_views):
            angle = (i / num_views) * 2 * np.pi

            # Camera position
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 2.0  # Height

            # Look-at matrix (simplified)
            camera_matrix = np.array([
                [1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, z]
            ])
            camera_matrices.append(camera_matrix)

            # View angles
            view_angle = [
                np.degrees(angle),  # azimuth
                15.0,  # elevation
                0.0,   # roll
                60.0,  # fov_h
                45.0,  # fov_v
                radius  # distance
            ]
            view_angles.append(view_angle)

        return (torch.tensor(camera_matrices, dtype=torch.float32),
                torch.tensor(view_angles, dtype=torch.float32))

    @torch.no_grad()
    def analyze_scene(self, image_paths: List[str]) -> Dict:
        """
        Analyze a 3D scene from multiple views

        Args:
            image_paths: List of paths to multi-view images

        Returns:
            Dictionary with analysis results
        """
        # Preprocess images
        images = self.preprocess_images(image_paths)
        num_views = len(images)

        # Generate camera parameters
        camera_matrices, view_angles = self.generate_camera_matrices(num_views)

        # Batch and move to device
        image_batch = torch.stack(images).unsqueeze(0).to(self.device)
        camera_matrices = camera_matrices.unsqueeze(0).to(self.device)
        view_angles = view_angles.unsqueeze(0).to(self.device)

        # Split images for model input
        images_list = [image_batch[:, i] for i in range(num_views)]

        # Forward pass
        outputs = self.model.encode_images(images_list, camera_matrices, view_angles)

        results = {
            'num_views': num_views,
            'depth_maps': [],
            'coordinates': None,
            'confidence': None,
            'voxels': None,
            'scene_description': ""
        }

        # Extract depth maps
        if 'depth_maps' in outputs:
            for depth in outputs['depth_maps']:
                depth_np = depth.cpu().numpy()[0, 0]
                results['depth_maps'].append(depth_np)

        # Extract 3D coordinates
        if 'coordinates' in outputs:
            coords = outputs['coordinates'].cpu().numpy()[0]
            confidence = outputs['confidence'].cpu().numpy()[0]
            results['coordinates'] = coords
            results['confidence'] = confidence

        # Extract voxel reconstruction
        if 'voxels' in outputs:
            voxels = outputs['voxels'].cpu().numpy()[0, 0]
            results['voxels'] = voxels

        # Generate scene description
        results['scene_description'] = self.generate_description(outputs)

        return results

    def generate_description(self, outputs: Dict) -> str:
        """Generate natural language description of the scene"""
        # This would use the language model component in production
        # For now, return a structured description based on outputs

        description = "3D Scene Analysis:\n"

        if 'coordinates' in outputs:
            num_objects = outputs['coordinates'].shape[1]
            description += f"- Detected {num_objects} potential objects in 3D space\n"

        if 'depth_maps' in outputs:
            num_depth = len(outputs['depth_maps'])
            description += f"- Generated depth maps for {num_depth} views\n"

        if 'voxels' in outputs:
            voxel_occupancy = (outputs['voxels'] > 0.5).float().mean().item()
            description += f"- Voxel occupancy: {voxel_occupancy:.2%}\n"

        description += "- Scene suitable for gaming/metaverse applications\n"

        return description

    def visualize_results(self, results: Dict, save_path: Optional[str] = None):
        """Visualize analysis results"""
        fig = plt.figure(figsize=(15, 10))

        # Plot depth maps
        num_depth = len(results['depth_maps'])
        for i, depth in enumerate(results['depth_maps'][:4]):  # Show max 4 views
            ax = fig.add_subplot(3, 4, i + 1)
            im = ax.imshow(depth, cmap='viridis')
            ax.set_title(f'Depth View {i + 1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Plot 3D coordinates if available
        if results['coordinates'] is not None:
            ax = fig.add_subplot(3, 2, 3, projection='3d')
            coords = results['coordinates']
            confidence = results['confidence'].squeeze()

            # Filter by confidence
            mask = confidence > 0.5
            coords_filtered = coords[mask]

            if len(coords_filtered) > 0:
                ax.scatter(coords_filtered[:, 0],
                         coords_filtered[:, 1],
                         coords_filtered[:, 2],
                         c=confidence[mask],
                         cmap='coolwarm',
                         s=50)
                ax.set_title('3D Object Positions')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

        # Plot voxel slice if available
        if results['voxels'] is not None:
            ax = fig.add_subplot(3, 2, 4)
            voxel_slice = results['voxels'][32]  # Middle slice
            ax.imshow(voxel_slice, cmap='binary')
            ax.set_title('Voxel Reconstruction (Z=32)')
            ax.axis('off')

        # Add text description
        ax = fig.add_subplot(3, 1, 3)
        ax.text(0.1, 0.5, results['scene_description'],
               fontsize=10, verticalalignment='center')
        ax.set_title('Scene Analysis')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.show()

    def export_point_cloud(self, results: Dict, save_path: str):
        """Export results as 3D point cloud"""
        if results['coordinates'] is None:
            print("No 3D coordinates to export")
            return

        coords = results['coordinates']
        confidence = results['confidence'].squeeze()

        # Filter by confidence
        mask = confidence > 0.3
        points = coords[mask]

        if len(points) == 0:
            print("No confident points to export")
            return

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Color by confidence
        colors = plt.cm.coolwarm(confidence[mask])[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save point cloud
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Exported point cloud to {save_path}")

    def export_voxels(self, results: Dict, save_path: str):
        """Export voxel grid as mesh"""
        if results['voxels'] is None:
            print("No voxels to export")
            return

        voxels = results['voxels']

        # Threshold voxels
        occupied = voxels > 0.5

        # Extract surface mesh using marching cubes
        from skimage import measure
        vertices, faces, normals, values = measure.marching_cubes(
            occupied, level=0.5
        )

        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        # Save mesh
        o3d.io.write_triangle_mesh(save_path, mesh)
        print(f"Exported voxel mesh to {save_path}")

    def interactive_demo(self):
        """Run interactive demo with sample data"""
        print("Zen-3D Interactive Demo")
        print("-" * 50)

        # Generate sample multi-view data
        num_views = 4
        sample_images = []
        for i in range(num_views):
            # Create synthetic test image
            img = torch.randn(3, 224, 224)
            sample_images.append(img)

        print(f"Generated {num_views} synthetic views")

        # Run analysis
        print("Analyzing 3D scene...")
        camera_matrices, view_angles = self.generate_camera_matrices(num_views)

        image_batch = torch.stack(sample_images).unsqueeze(0).to(self.device)
        camera_matrices = camera_matrices.unsqueeze(0).to(self.device)
        view_angles = view_angles.unsqueeze(0).to(self.device)

        images_list = [image_batch[:, i] for i in range(num_views)]

        with torch.no_grad():
            outputs = self.model.encode_images(images_list, camera_matrices, view_angles)

        # Generate interactive description
        print("\n" + "=" * 50)
        print("3D SCENE ANALYSIS RESULTS")
        print("=" * 50)

        if 'depth_maps' in outputs:
            print(f"✓ Depth estimation: {len(outputs['depth_maps'])} depth maps generated")

        if 'coordinates' in outputs:
            coords = outputs['coordinates']
            print(f"✓ 3D localization: {coords.shape[1]} anchor points detected")

        if 'voxels' in outputs:
            voxels = outputs['voxels']
            occupancy = (voxels > 0.5).float().mean().item()
            print(f"✓ Voxel reconstruction: {occupancy:.1%} space occupancy")

        print("\nPotential Gaming/Metaverse Applications:")
        print("• Environment mapping for VR/AR experiences")
        print("• Collision detection for game physics")
        print("• Procedural level generation from real scenes")
        print("• NFT-based 3D asset creation")
        print("• Spatial anchoring for multiplayer interactions")

        return outputs


class Zoo3DApplications:
    """
    Specific applications for Zoo Labs ecosystem
    Gaming, NFT, and metaverse focused utilities
    """

    def __init__(self, model_path: Optional[str] = None):
        self.inference = Zen3DInference(model_path)

    def nft_asset_generator(self, image_paths: List[str]) -> Dict:
        """Generate 3D NFT assets from multi-view images"""
        results = self.inference.analyze_scene(image_paths)

        nft_data = {
            'asset_type': '3D_MODEL',
            'metadata': {
                'views': len(image_paths),
                'vertices': 0,
                'faces': 0,
                'materials': []
            },
            'rarity_traits': [],
            'game_stats': {}
        }

        # Extract 3D model data
        if results['voxels'] is not None:
            voxels = results['voxels']
            occupied = (voxels > 0.5).sum()
            nft_data['metadata']['voxels'] = int(occupied)

            # Rarity based on complexity
            if occupied < 1000:
                nft_data['rarity_traits'].append('MINIMALIST')
            elif occupied > 10000:
                nft_data['rarity_traits'].append('COMPLEX')

        # Game statistics
        if results['coordinates'] is not None:
            num_objects = results['coordinates'].shape[0]
            nft_data['game_stats'] = {
                'interaction_points': num_objects,
                'complexity_score': min(100, num_objects * 10),
                'spatial_volume': float(results['coordinates'].std())
            }

        return nft_data

    def metaverse_scene_builder(self, image_paths: List[str]) -> Dict:
        """Build metaverse-ready scene from images"""
        results = self.inference.analyze_scene(image_paths)

        scene = {
            'format': 'ZOO_SCENE_V1',
            'components': [],
            'lighting': {},
            'physics': {},
            'interactions': []
        }

        # Add depth-based terrain
        if results['depth_maps']:
            scene['components'].append({
                'type': 'TERRAIN',
                'data': 'depth_mesh',
                'lod_levels': 3
            })

        # Add detected objects
        if results['coordinates'] is not None:
            for i, (coord, conf) in enumerate(zip(results['coordinates'],
                                                 results['confidence'])):
                if conf > 0.5:
                    scene['components'].append({
                        'type': 'OBJECT',
                        'id': f'obj_{i}',
                        'position': coord.tolist(),
                        'interactive': conf > 0.7
                    })

        # Physics configuration
        if results['voxels'] is not None:
            scene['physics'] = {
                'collision_mesh': 'voxel_simplified',
                'gravity': True,
                'static_objects': len(scene['components'])
            }

        return scene

    def gaming_level_analyzer(self, image_paths: List[str]) -> Dict:
        """Analyze images for gaming level design"""
        results = self.inference.analyze_scene(image_paths)

        level_analysis = {
            'layout_type': '',
            'traversability': 0.0,
            'cover_points': [],
            'spawn_locations': [],
            'objectives': []
        }

        # Analyze spatial layout
        if results['depth_maps']:
            avg_depth = np.mean([d.mean() for d in results['depth_maps']])
            if avg_depth < 3.0:
                level_analysis['layout_type'] = 'CLOSE_QUARTERS'
            elif avg_depth > 7.0:
                level_analysis['layout_type'] = 'OPEN_FIELD'
            else:
                level_analysis['layout_type'] = 'MIXED'

        # Find strategic positions
        if results['coordinates'] is not None:
            coords = results['coordinates']
            confidence = results['confidence'].squeeze()

            # High confidence points as cover
            cover_mask = confidence > 0.7
            level_analysis['cover_points'] = coords[cover_mask].tolist()

            # Low density areas as spawn points
            if len(coords) > 0:
                # Simple clustering for spawn locations
                spawn_candidates = []
                for i in range(0, len(coords), 3):
                    spawn_candidates.append(coords[i].tolist())
                level_analysis['spawn_locations'] = spawn_candidates[:4]

        # Traversability score
        if results['voxels'] is not None:
            free_space = 1.0 - (results['voxels'] > 0.5).mean()
            level_analysis['traversability'] = float(free_space)

        return level_analysis


def main():
    """Main demonstration"""
    print("Zen-3D Model - Advanced Spatial Understanding")
    print("=" * 60)

    # Initialize inference
    inference = Zen3DInference()

    # Run interactive demo
    outputs = inference.interactive_demo()

    # Initialize Zoo applications
    zoo_apps = Zoo3DApplications()

    # Demo NFT generation
    print("\n" + "=" * 60)
    print("NFT ASSET GENERATION DEMO")
    print("=" * 60)
    sample_paths = ["view1.jpg", "view2.jpg", "view3.jpg", "view4.jpg"]
    nft_data = zoo_apps.nft_asset_generator(sample_paths)
    print(json.dumps(nft_data, indent=2))

    # Demo metaverse scene
    print("\n" + "=" * 60)
    print("METAVERSE SCENE BUILDER DEMO")
    print("=" * 60)
    scene_data = zoo_apps.metaverse_scene_builder(sample_paths)
    print(f"Generated scene with {len(scene_data['components'])} components")

    # Demo gaming analysis
    print("\n" + "=" * 60)
    print("GAMING LEVEL ANALYSIS DEMO")
    print("=" * 60)
    level_data = zoo_apps.gaming_level_analyzer(sample_paths)
    print(f"Level type: {level_data['layout_type']}")
    print(f"Traversability: {level_data['traversability']:.1%}")


if __name__ == "__main__":
    main()