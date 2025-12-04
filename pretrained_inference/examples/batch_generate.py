#!/usr/bin/env python3
"""
Batch generation script for testing multiple prompts.

This script allows you to test multiple prompts at once and compare results.

Usage:
    python batch_generate.py --model pointe
    python batch_generate.py --model shape --prompts prompts.txt
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pointe_inference import PointEInference
from models.shape_inference import ShapEInference
from utils.visualization import save_point_cloud_ply


DEFAULT_PROMPTS = [
    "a red car",
    "a blue chair",
    "a green apple",
    "a wooden table",
    "a coffee mug",
]


def main():
    parser = argparse.ArgumentParser(description='Batch generate 3D objects')
    parser.add_argument(
        '--model',
        choices=['pointe', 'shape'],
        default='pointe',
        help='Model to use'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        help='Path to text file with prompts (one per line)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/batch',
        help='Directory to save outputs'
    )

    args = parser.parse_args()

    # Load prompts
    if args.prompts:
        with open(args.prompts, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS
        print(f"Using default prompts: {prompts}")

    # Initialize model
    print("=" * 60)
    print(f"Batch Generation with {args.model.upper()}")
    print("=" * 60)

    if args.model == 'pointe':
        model = PointEInference()
        model.load_models(use_upsampler=True, guidance_scale=3.0)
    else:
        model = ShapEInference()
        model.load_model(model_type='text300M')

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate for each prompt
    for i, prompt in enumerate(prompts):
        print("\n" + "=" * 60)
        print(f"Prompt {i+1}/{len(prompts)}: {prompt}")
        print("=" * 60)

        try:
            if args.model == 'pointe':
                point_clouds = model.generate_from_text(prompt, num_samples=1)
                for j, pc in enumerate(point_clouds):
                    output_path = output_dir / f"{i:02d}_{prompt.replace(' ', '_')}.ply"
                    save_point_cloud_ply(pc, str(output_path))

            else:  # shape
                latents = model.generate_from_text(prompt, num_samples=1)
                for j, latent in enumerate(latents):
                    output_path = output_dir / f"{i:02d}_{prompt.replace(' ', '_')}.ply"
                    model.latent_to_mesh(latent, str(output_path))

        except Exception as e:
            print(f"Error generating '{prompt}': {e}")
            continue

    print("\n" + "=" * 60)
    print(f"Done! Generated objects for {len(prompts)} prompts")
    print(f"Outputs saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
