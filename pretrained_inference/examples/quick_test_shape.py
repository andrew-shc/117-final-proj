#!/usr/bin/env python3
"""
Quick test script for Shap-E pretrained model.

This script demonstrates how to quickly generate 3D meshes and point clouds
from text using OpenAI's pretrained Shap-E model.

Usage:
    python quick_test_shape.py
    python quick_test_shape.py --prompt "a donut"
    python quick_test_shape.py --prompt "an avocado armchair" --samples 2
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.shape_inference import ShapEInference


def main():
    parser = argparse.ArgumentParser(description='Generate 3D objects with Shap-E')
    parser.add_argument(
        '--prompt',
        type=str,
        default='a shark',
        help='Text description of the 3D object'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=15.0,
        help='Classifier-free guidance scale (higher = more faithful to prompt)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=64,
        help='Number of diffusion steps'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/shape',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--mesh-only',
        action='store_true',
        help='Only generate meshes (skip point clouds)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render views of the generated objects'
    )

    args = parser.parse_args()

    # Initialize model
    print("=" * 60)
    print("Shap-E Quick Test")
    print("=" * 60)

    model = ShapEInference()
    model.load_model(model_type='text300M')

    # Generate latents
    print("\n" + "=" * 60)
    print("Generating...")
    print("=" * 60)

    latents = model.generate_from_text(
        prompt=args.prompt,
        num_samples=args.samples,
        guidance_scale=args.guidance_scale,
        num_steps=args.steps
    )

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Saving outputs...")
    print("=" * 60)

    for i, latent in enumerate(latents):
        # Save mesh
        mesh_path = output_dir / f"sample_{i:02d}.ply"
        model.latent_to_mesh(latent, str(mesh_path))

        # Optionally render views
        if args.render:
            render_dir = output_dir / f"sample_{i:02d}_renders"
            model.render_latent(latent, str(render_dir))

        # Optionally save point cloud
        if not args.mesh_only:
            pc = model.latent_to_point_cloud(latent, num_points=4096)
            pc_path = output_dir / f"sample_{i:02d}_pointcloud.ply"
            model.save_point_cloud(pc, str(pc_path))

    print("\n" + "=" * 60)
    print(f"Done! Generated {len(latents)} object(s)")
    print(f"Outputs saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
