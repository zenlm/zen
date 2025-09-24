#!/usr/bin/env python3
"""
Example usage of BitDelta for personalizing Zen models.

This script demonstrates:
1. Creating personalized profiles from examples
2. Switching between profiles
3. Generating with different styles
4. Profile compression statistics
"""

import torch
from pathlib import Path
from zen_integration import ZenPersonalizationManager, ZenBitDelta
from bitdelta import BitDeltaConfig


def main():
    """Demonstrate BitDelta personalization."""
    
    print("=" * 60)
    print("BitDelta: Personalized LLM Demo")
    print("=" * 60)
    
    # Initialize manager with zen-nano for efficiency
    print("\n1. Initializing Zen-Nano with BitDelta...")
    manager = ZenPersonalizationManager('zen-nano-4b')
    
    # Example training data for different styles
    technical_examples = [
        "The asymptotic complexity of the algorithm is O(n log n) due to the divide-and-conquer approach employed in the merge sort implementation.",
        "Memory allocation follows a buddy system pattern, ensuring logarithmic fragmentation with constant-time coalescing operations.",
        "The distributed consensus protocol achieves Byzantine fault tolerance through a three-phase commit mechanism with cryptographic verification.",
        "Cache coherence is maintained via the MESI protocol, minimizing inter-processor communication overhead in multicore architectures.",
        "The type system enforces referential transparency through monadic composition and algebraic data types."
    ]
    
    creative_examples = [
        "The moonlight danced across the water like silver serpents, weaving tales of ancient mysteries beneath the surface.",
        "Her laughter was a symphony of wind chimes, each note painting colors in the air that only dreamers could see.",
        "The old library whispered secrets through dusty pages, each book a portal to worlds unbound by time or logic.",
        "Thoughts cascaded like autumn leaves, swirling in patterns that defied the gravity of reason.",
        "In the garden of possibilities, every flower bloomed with the fragrance of untold stories."
    ]
    
    casual_examples = [
        "Hey, just wanted to check in and see how things are going with the project!",
        "That's totally fine! We can definitely work around that schedule.",
        "Honestly, I think the simpler approach might be better here.",
        "Sure thing! Let me know if you need any help with that.",
        "Yeah, I ran into the same issue yesterday. Here's what worked for me..."
    ]
    
    # Create personalization profiles
    print("\n2. Creating personalization profiles...")
    print("   - Technical style")
    manager.create_profile_from_examples('technical', technical_examples, 'technical writing')
    
    print("   - Creative style")
    manager.create_profile_from_examples('creative', creative_examples, 'creative writing')
    
    print("   - Casual style")
    manager.create_profile_from_examples('casual', casual_examples, 'casual conversation')
    
    # Demonstrate generation with different profiles
    print("\n3. Testing generation with different profiles...")
    
    test_prompt = "Explain how neural networks learn:"
    
    print(f"\nPrompt: {test_prompt}")
    print("-" * 40)
    
    # Technical style
    print("\n[Technical Profile]")
    response = manager.generate_with_profile(
        test_prompt, 
        profile_name='technical',
        max_length=150
    )
    print(response)
    
    # Creative style
    print("\n[Creative Profile]")
    response = manager.generate_with_profile(
        test_prompt,
        profile_name='creative', 
        max_length=150
    )
    print(response)
    
    # Casual style
    print("\n[Casual Profile]")
    response = manager.generate_with_profile(
        test_prompt,
        profile_name='casual',
        max_length=150
    )
    print(response)
    
    # Demonstrate profile merging
    print("\n4. Creating hybrid profile...")
    hybrid_name = manager.merge_profiles(
        ['technical', 'creative'],
        weights=[0.6, 0.4],
        new_name='tech_creative_hybrid'
    )
    
    print(f"   Created hybrid: {hybrid_name}")
    print("\n[Hybrid Profile (60% technical, 40% creative)]")
    response = manager.generate_with_profile(
        test_prompt,
        profile_name=hybrid_name,
        max_length=150
    )
    print(response)
    
    # Show compression statistics
    print("\n5. Compression Statistics")
    print("-" * 40)
    
    for profile_name in ['technical', 'creative', 'casual', hybrid_name]:
        profile = manager.bitdelta.profiles[profile_name]
        
        # Calculate sizes
        num_params = sum(s.numel() for s, _ in profile['deltas'].values())
        storage_mb = sum(
            s.numel() / 8 / 1024 / 1024 + sc.numel() * 4 / 1024 / 1024
            for s, sc in profile['deltas'].values()
        )
        
        print(f"\nProfile: {profile_name}")
        print(f"  Compression ratio: {profile['compression_ratio']:.1f}x")
        print(f"  Parameters: {num_params:,}")
        print(f"  Storage size: {storage_mb:.2f} MB")
    
    # Export profiles for sharing
    print("\n6. Exporting profiles...")
    export_dir = Path("exported_profiles")
    export_dir.mkdir(exist_ok=True)
    
    for profile_name in ['technical', 'creative', 'casual']:
        export_path = export_dir / f"{profile_name}.bitdelta"
        manager.export_profile(profile_name, str(export_path))
        print(f"   Exported: {export_path}")
    
    # Demonstrate profile switching speed
    print("\n7. Profile Switching Performance")
    print("-" * 40)
    
    import time
    
    profiles = ['technical', 'creative', 'casual']
    for profile in profiles:
        start_time = time.time()
        manager.activate_profile(profile)
        switch_time = (time.time() - start_time) * 1000
        print(f"   {profile}: {switch_time:.1f}ms")
    
    print("\n" + "=" * 60)
    print("BitDelta Demo Complete!")
    print("=" * 60)


def advanced_example():
    """Advanced usage with custom configuration."""
    
    print("\nAdvanced BitDelta Configuration")
    print("-" * 40)
    
    # Custom configuration for optimal quality/compression trade-off
    config = BitDeltaConfig(
        block_size=64,  # Smaller blocks for finer control
        use_per_channel_scale=True,  # Better quality
        scale_bits=16,  # 16-bit scale factors
        straight_through=True,  # STE for training
        delta_regularization=0.005,  # Moderate regularization
        sparsity_penalty=0.0001,  # Encourage sparsity
        learning_rate_scale=0.2,  # Slower learning for deltas
        warmup_steps=200  # Longer warmup
    )
    
    # Initialize with custom config
    bitdelta = ZenBitDelta('zen-nano-4b', config=config)
    bitdelta.load_base_model()
    
    # Apply BitDelta to specific layers only
    target_layers = ['q_proj', 'v_proj', 'down_proj']  # Only attention and down projection
    bitdelta.apply_bitdelta_layers(target_modules=target_layers)
    
    print(f"Applied BitDelta to layers: {target_layers}")
    
    # Custom training data
    domain_specific_data = [
        {"text": "The quantum entanglement phenomenon exhibits non-local correlations."},
        {"text": "Hilbert space formalism provides the mathematical foundation."},
        {"text": "Wave function collapse occurs upon measurement interaction."},
    ]
    
    # Train with custom parameters
    bitdelta.personalize(
        profile_name='quantum_physics',
        train_data=domain_specific_data,
        learning_rate=1e-5,  # Lower learning rate
        epochs=5,  # More epochs for small dataset
        batch_size=1  # Small batch for few examples
    )
    
    print("Created specialized quantum physics profile")
    
    # Save to disk with compression
    bitdelta.save_to_disk('bitdelta_profiles')
    print("Saved profiles to disk")
    
    # Load from disk (simulating restart)
    bitdelta2 = ZenBitDelta('zen-nano-4b')
    bitdelta2.load_from_disk('bitdelta_profiles')
    print("Loaded profiles from disk")
    
    # Use loaded profile
    model = bitdelta2.load_profile('quantum_physics')
    print("Activated quantum physics personalization")


def compression_analysis():
    """Analyze compression ratios for different model sizes."""
    
    print("\nCompression Analysis")
    print("-" * 40)
    
    models = [
        ('zen-nano-4b', 4e9, 16),  # 4B params, 16GB
        ('zen-next-7b', 7e9, 28),  # 7B params, 28GB
        ('zen-coder-14b', 14e9, 56),  # 14B params, 56GB
        ('zen-omni-30b', 30e9, 120),  # 30B params, 120GB
    ]
    
    print("\nModel Compression Ratios:")
    print(f"{'Model':<15} {'Params':<10} {'Original':<12} {'BitDelta':<12} {'Ratio':<8}")
    print("-" * 60)
    
    for model_name, num_params, size_gb in models:
        # Estimate BitDelta size
        # 1 bit per param + scales (assume 1% overhead)
        bitdelta_size_gb = (num_params / 8 / 1e9) * 1.01
        ratio = size_gb / bitdelta_size_gb
        
        print(f"{model_name:<15} {num_params/1e9:.1f}B{'':<5} "
              f"{size_gb:.0f}GB{'':<8} {bitdelta_size_gb:.2f}GB{'':<8} "
              f"{ratio:.0f}x")
    
    print("\nStorage Requirements for 100 Users:")
    print(f"{'Model':<15} {'Traditional':<15} {'BitDelta':<15} {'Savings':<10}")
    print("-" * 60)
    
    for model_name, num_params, size_gb in models:
        traditional = size_gb * 100  # 100 full copies
        bitdelta_size_gb = (num_params / 8 / 1e9) * 1.01
        bitdelta_total = size_gb + (bitdelta_size_gb * 100)  # Base + 100 deltas
        savings = (1 - bitdelta_total / traditional) * 100
        
        print(f"{model_name:<15} {traditional:.0f}GB{'':<10} "
              f"{bitdelta_total:.1f}GB{'':<10} {savings:.1f}%")


if __name__ == "__main__":
    # Note: This example requires PyTorch and transformers to be installed
    # The actual execution would need the Zen models to be available
    
    print("""
    Note: This example demonstrates the BitDelta API.
    Actual execution requires:
    1. PyTorch and transformers installed
    2. Access to Zen model weights
    3. GPU for larger models (recommended)
    
    The code shows how to:
    - Create personalized profiles
    - Switch between styles
    - Merge profiles
    - Export/import profiles
    - Analyze compression ratios
    """)
    
    # Uncomment to run with actual models:
    # main()
    # advanced_example()
    
    # This can run without models:
    compression_analysis()