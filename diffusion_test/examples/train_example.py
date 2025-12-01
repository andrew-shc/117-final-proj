"""
Example: Training a diffusion model on 3D point clouds

This script demonstrates how to:
1. Create synthetic training data (spheres, cubes, torus)
2. Train a diffusion model from scratch
3. Save checkpoints
4. Monitor training progress
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from models import PointCloudDiffusionModel
from utils import (
    generate_sphere_point_cloud,
    generate_cube_point_cloud,
    generate_torus_point_cloud,
    normalize_point_cloud,
    visualize_multiple_point_clouds
)


class SyntheticPointCloudDataset(Dataset):
    """
    Synthetic point cloud dataset with multiple shapes
    """
    def __init__(self, num_samples=1000, num_points=1024, shapes=['sphere', 'cube', 'torus']):
        self.num_samples = num_samples
        self.num_points = num_points
        self.shapes = shapes
        self.data = []

        print(f"Generating {num_samples} synthetic point clouds...")
        for i in tqdm(range(num_samples)):
            shape_type = np.random.choice(shapes)

            if shape_type == 'sphere':
                pc = generate_sphere_point_cloud(num_points, radius=np.random.uniform(0.8, 1.2))
            elif shape_type == 'cube':
                pc = generate_cube_point_cloud(num_points, size=np.random.uniform(1.5, 2.5))
            elif shape_type == 'torus':
                pc = generate_torus_point_cloud(num_points, R=1.0, r=np.random.uniform(0.2, 0.4))

            # Normalize
            pc = normalize_point_cloud(pc, method='sphere')

            # Add small random noise
            pc += np.random.randn(*pc.shape) * 0.02

            self.data.append(pc.astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        batch = batch.to(device)

        # Forward pass
        loss = model(batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        loss = model(batch)
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved to {filepath}")


@torch.no_grad()
def generate_samples_during_training(model, device, save_path, num_samples=4):
    """Generate samples during training to monitor progress"""
    model.eval()
    samples = model.sample(batch_size=num_samples, device=device)

    visualize_multiple_point_clouds(
        [samples[i] for i in range(num_samples)],
        [f"Sample {i+1}" for i in range(num_samples)],
        save_path=save_path,
        show=False
    )
    print(f"Generated samples saved to {save_path}")


def main():
    # Hyperparameters
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_TRAIN_SAMPLES = 5000
    NUM_VAL_SAMPLES = 500
    NUM_POINTS = 1024

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'pretrained'
    )
    os.makedirs(output_dir, exist_ok=True)

    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = SyntheticPointCloudDataset(
        num_samples=NUM_TRAIN_SAMPLES,
        num_points=NUM_POINTS,
        shapes=['sphere', 'cube', 'torus']
    )

    print("\nCreating validation dataset...")
    val_dataset = SyntheticPointCloudDataset(
        num_samples=NUM_VAL_SAMPLES,
        num_points=NUM_POINTS,
        shapes=['sphere', 'cube', 'torus']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Create model
    print("\nInitializing model...")
    model = PointCloudDiffusionModel(
        num_points=NUM_POINTS,
        point_dim=3,
        hidden_dim=128,
        num_timesteps=1000
    )
    model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)

    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Training loss: {train_loss:.4f}")

        # Validate
        val_loss = evaluate(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)

            # Generate samples to visualize progress
            samples_path = os.path.join(output_dir, f'samples_epoch_{epoch}.png')
            generate_samples_during_training(model, device, samples_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, 'point_cloud_diffusion.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            print(f"New best model! Validation loss: {val_loss:.4f}")

    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == '__main__':
    main()
