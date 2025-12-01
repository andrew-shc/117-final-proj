"""
3D Point Cloud Diffusion Model

A complete implementation of diffusion models for 3D point cloud generation,
based on Denoising Diffusion Probabilistic Models (DDPM).

Main components:
- models: Point cloud diffusion model implementations
- utils: Utilities for point cloud processing and visualization
- examples: Example scripts for training and inference
"""

__version__ = '1.0.0'

from .models import PointCloudDiffusionModel, PointNet
from .utils import (
    generate_sphere_point_cloud,
    generate_cube_point_cloud,
    normalize_point_cloud,
    visualize_point_cloud,
    save_point_cloud,
    load_point_cloud
)

__all__ = [
    'PointCloudDiffusionModel',
    'PointNet',
    'generate_sphere_point_cloud',
    'generate_cube_point_cloud',
    'normalize_point_cloud',
    'visualize_point_cloud',
    'save_point_cloud',
    'load_point_cloud',
]
