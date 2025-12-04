#!/usr/bin/env python3
"""
Quick diagnostic to check sampling speed and iterations.
"""

import time

try:
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.models.download import load_checkpoint
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    import torch

    print("Point-E is installed!")
    print("\nChecking default configurations...")

    # Check base config
    base_config = DIFFUSION_CONFIGS['base40M-textvec']
    print(f"Base model timesteps: {base_config.get('timesteps', 'N/A')}")

    # Check if there are sampling step configs
    print("\nNote: Point-E uses Karras sampling by default with ~64 steps per stage")
    print("This is different from the 1024 timesteps in the diffusion schedule")
    print("\nWith upsampler: ~64 steps (base) + ~64 steps (upsampler) = ~128 total iterations")
    print("Without upsampler: ~64 steps")

except ImportError as e:
    print(f"Point-E not installed: {e}")
    print("\nInstall with: pip install git+https://github.com/openai/point-e.git")
