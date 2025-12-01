"""
Example: Using a pre-trained diffusion model for point cloud generation

This script demonstrates how to:
1. Load a pre-trained diffusion model
2. Generate new point clouds from noise
3. Interpolate between two point clouds
4. Visualize the results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models import PointCloudDiffusionModel
from utils import (
    generate_sphere_point_cloud,
    generate_torus_point_cloud,
    normalize_point_cloud,
    visualize_point_cloud,
    visualize_multiple_point_clouds
)


def load_pretrained_model(model_path: str, device: str = 'cpu'):
    """Load a pre-trained diffusion model"""
    model = PointCloudDiffusionModel(
        num_points=1024,
        point_dim=3,
        hidden_dim=128,
        num_timesteps=1000
    )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pre-trained model from {model_path}")
        print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Training loss: {checkpoint.get('loss', 'unknown'):.4f}")
    else:
        print(f"Warning: Pre-trained model not found at {model_path}")
        print("Using randomly initialized model for demonstration")

    model.to(device)
    model.eval()
    return model


def generate_samples(model, num_samples=4, device='cpu'):
    """Generate point cloud samples from noise"""
    print(f"\nGenerating {num_samples} point cloud samples...")

    with torch.no_grad():
        samples = model.sample(batch_size=num_samples, device=device)

    print(f"Generated samples shape: {samples.shape}")
    return samples


def interpolate_point_clouds(model, pc1, pc2, num_steps=5, device='cpu'):
    """Interpolate between two point clouds"""
    print(f"\nInterpolating between two point clouds with {num_steps} steps...")

    pc1_tensor = torch.from_numpy(pc1).unsqueeze(0).float().to(device)
    pc2_tensor = torch.from_numpy(pc2).unsqueeze(0).float().to(device)

    interpolations = []
    lambdas = np.linspace(0, 1, num_steps)

    for lambda_ in lambdas:
        with torch.no_grad():
            interp = model.interpolate(pc1_tensor, pc2_tensor, t=500, lambda_=lambda_)
        interpolations.append(interp.cpu())

    return interpolations


def denoise_example(model, clean_pc, noise_level=0.5, device='cpu'):
    """Example of denoising a point cloud"""
    print(f"\nDenoising example with noise level {noise_level}...")

    # Add noise
    noise = np.random.randn(*clean_pc.shape) * noise_level
    noisy_pc = clean_pc + noise

    # Convert to tensor
    noisy_tensor = torch.from_numpy(noisy_pc).unsqueeze(0).float().to(device)

    # Denoise by running reverse diffusion from a specific timestep
    with torch.no_grad():
        t = int(model.num_timesteps * 0.3)  # Start from 30% of the way
        for step in reversed(range(t)):
            noisy_tensor = model.p_sample(noisy_tensor, step)

    denoised = noisy_tensor.cpu().numpy()[0]
    return noisy_pc, denoised


def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Path to pre-trained model
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'pretrained',
        'point_cloud_diffusion.pth'
    )

    # Load model
    model = load_pretrained_model(model_path, device=device)

    # ===== Example 1: Generate samples =====
    print("\n" + "="*50)
    print("Example 1: Generating point clouds from noise")
    print("="*50)

    samples = generate_samples(model, num_samples=4, device=device)

    # Visualize generated samples
    visualize_multiple_point_clouds(
        [samples[i] for i in range(4)],
        [f"Generated Sample {i+1}" for i in range(4)],
        save_path='generated_samples.png',
        show=False
    )
    print("Saved visualization to: generated_samples.png")

    # ===== Example 2: Interpolation =====
    print("\n" + "="*50)
    print("Example 2: Interpolating between point clouds")
    print("="*50)

    # Create two different shapes
    sphere = generate_sphere_point_cloud(num_points=1024, radius=1.0)
    sphere = normalize_point_cloud(sphere, method='sphere')

    torus = generate_torus_point_cloud(num_points=1024, R=1.0, r=0.3)
    torus = normalize_point_cloud(torus, method='sphere')

    # Interpolate
    interpolations = interpolate_point_clouds(
        model, sphere, torus, num_steps=5, device=device
    )

    # Visualize interpolation
    titles = ['Sphere'] + [f'Step {i+1}' for i in range(3)] + ['Torus']
    visualize_multiple_point_clouds(
        [torch.from_numpy(sphere)] + interpolations[1:-1] + [torch.from_numpy(torus)],
        titles,
        save_path='interpolation.png',
        show=False,
        ncols=5
    )
    print("Saved interpolation visualization to: interpolation.png")

    # ===== Example 3: Denoising =====
    print("\n" + "="*50)
    print("Example 3: Denoising a point cloud")
    print("="*50)

    # Create a clean sphere
    clean_sphere = generate_sphere_point_cloud(num_points=1024, radius=1.0, noise=0.0)
    clean_sphere = normalize_point_cloud(clean_sphere, method='sphere')

    # Add noise and denoise
    noisy_pc, denoised_pc = denoise_example(model, clean_sphere, noise_level=0.3, device=device)

    # Visualize
    visualize_multiple_point_clouds(
        [clean_sphere, noisy_pc, denoised_pc],
        ['Original', 'Noisy', 'Denoised'],
        save_path='denoising.png',
        show=False,
        ncols=3
    )
    print("Saved denoising visualization to: denoising.png")

    # ===== Example 4: Single sample visualization =====
    print("\n" + "="*50)
    print("Example 4: High-quality single sample")
    print("="*50)

    single_sample = generate_samples(model, num_samples=1, device=device)
    visualize_point_cloud(
        single_sample[0],
        title="Generated Point Cloud (High Quality)",
        save_path='single_sample.png',
        show=False,
        point_size=2.0
    )
    print("Saved single sample visualization to: single_sample.png")

    print("\n" + "="*50)
    print("All examples completed successfully!")
    print("="*50)
    print("\nGenerated files:")
    print("  - generated_samples.png")
    print("  - interpolation.png")
    print("  - denoising.png")
    print("  - single_sample.png")


if __name__ == '__main__':
    main()
