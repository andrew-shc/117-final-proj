#!/usr/bin/env python3
"""
Quick test script for Point-E pretrained model.

This script demonstrates how to quickly generate 3D point clouds from text
using OpenAI's pretrained Point-E model.

Usage:
    python quick_test_pointe.py
    python quick_test_pointe.py --prompt "a red car"
    python quick_test_pointe.py --prompt "a chair" --samples 3
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pointe_inference import PointEInference
from utils.visualization import (
    visualize_point_cloud_matplotlib,
    visualize_point_cloud_plotly,
    save_point_cloud_ply
)


def main():
    parser = argparse.ArgumentParser(description='Generate 3D point clouds with Point-E')
    parser.add_argument(
        '--prompt',
        type=str,
        default='a red motorcycle',
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
        default=3.0,
        help='Classifier-free guidance scale (higher = more faithful to prompt)'
    )
    parser.add_argument(
        '--no-upsampler',
        action='store_true',
        help='Skip upsampler (faster but lower resolution)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/pointe',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--visualize',
        choices=['matplotlib', 'plotly', 'both', 'none'],
        default='matplotlib',
        help='Visualization method'
    )

    args = parser.parse_args()

    # Initialize model
    print("=" * 60)
    print("Point-E Quick Test")
    print("=" * 60)

    model = PointEInference()
    model.load_models(
        use_upsampler=not args.no_upsampler,
        guidance_scale=args.guidance_scale
    )

    # Generate point clouds
    print("\n" + "=" * 60)
    print("Generating...")
    print("=" * 60)
    print(f"Guidance scale: {args.guidance_scale}")

    point_clouds = model.generate_from_text(
        prompt=args.prompt,
        num_samples=args.samples
    )

    # Save and visualize
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Saving outputs...")
    print("=" * 60)

    for i, pc in enumerate(point_clouds):
        # Save PLY
        ply_path = output_dir / f"sample_{i:02d}.ply"
        save_point_cloud_ply(pc, str(ply_path))

        # Visualize
        if args.visualize != 'none':
            if args.visualize in ['matplotlib', 'both']:
                img_path = output_dir / f"sample_{i:02d}_matplotlib.png"
                visualize_point_cloud_matplotlib(
                    pc,
                    output_path=str(img_path),
                    title=f"{args.prompt} (sample {i+1})"
                )

            if args.visualize in ['plotly', 'both']:
                html_path = output_dir / f"sample_{i:02d}_interactive.html"
                visualize_point_cloud_plotly(
                    pc,
                    output_path=str(html_path),
                    title=f"{args.prompt} (sample {i+1})"
                )

    print("\n" + "=" * 60)
    print(f"Done! Generated {len(point_clouds)} point cloud(s)")
    print(f"Outputs saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
