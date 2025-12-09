"""
Custom 3D Gaussian Splatting Training Script
"""

import argparse
import torch
import numpy as np
import pycolmap
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train Custom 3D Gaussian Splatting")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data directory with sparse/0 subdirectory")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    return parser.parse_args()


class GaussianSplattingTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iteration = 0
        
        self.output_path = Path(args.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Device: {self.device}")
        print(f"Output: {self.output_path}")
    
    def load_colmap_data(self):
        """Load COLMAP sparse reconstruction using pycolmap"""
        data_path = Path(self.args.data_path)
        sparse_path = data_path / "sparse" / "0"
        
        print(f"Loading COLMAP data from: {sparse_path}")
        
        # Load reconstruction
        reconstruction = pycolmap.Reconstruction(str(sparse_path))
        
        # Extract 3D points
        points3D = reconstruction.points3D
        num_points = len(points3D)
        
        if num_points > 0:
            self.xyz = np.zeros((num_points, 3))
            self.rgb = np.zeros((num_points, 3))
            
            for idx, (point_id, point) in enumerate(points3D.items()):
                self.xyz[idx] = point.xyz
                self.rgb[idx] = point.color
            
            print(f"Loaded {num_points} 3D points")
        else:
            print("Warning: No 3D points found")
            self.xyz, self.rgb = None, None
        
        # Store cameras and images
        self.cameras = reconstruction.cameras
        self.images = reconstruction.images
        
        print(f"Loaded {len(self.cameras)} cameras")
        print(f"Loaded {len(self.images)} images")
    
    def init_gaussians(self):
        """Initialize Gaussian parameters from point cloud"""
        if self.xyz is None:
            print("No point cloud data, skipping initialization")
            return
        
        print("Initializing Gaussians from point cloud...")
        num_points = len(self.xyz)
        
        # Convert to torch tensors
        self.positions = torch.tensor(self.xyz, dtype=torch.float32, device=self.device)
        self.colors = torch.tensor(self.rgb / 255.0, dtype=torch.float32, device=self.device)
        
        # Initialize other parameters
        self.scales = torch.ones((num_points, 3), device=self.device) * 0.01
        self.rotations = torch.zeros((num_points, 4), device=self.device)
        self.rotations[:, 0] = 1.0  # Identity quaternion
        self.opacities = torch.ones((num_points, 1), device=self.device) * 0.5
        
        print(f"Initialized {num_points} Gaussians")
    
    def train_iteration(self):
        """Single training iteration"""
        # TODO: Implement rendering and optimization
        pass
    
    def save_checkpoint(self, iteration):
        """Save checkpoint"""
        checkpoint_path = self.output_path / f"checkpoint_{iteration}.pth"
        torch.save({
            'iteration': iteration,
            'positions': self.positions,
            'colors': self.colors,
            'scales': self.scales,
            'rotations': self.rotations,
            'opacities': self.opacities,
        }, checkpoint_path)
        print(f"Saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        print("=" * 50)
        print("Starting training")
        print("=" * 50)
        
        # Load data
        self.load_colmap_data()
        
        # Initialize Gaussians
        self.init_gaussians()
        
        # Training loop
        for iteration in range(self.args.iterations):
            self.iteration = iteration
            
            # Training step
            self.train_iteration()
            
            # Logging
            if iteration % 100 == 0:
                print(f"Iteration {iteration}/{self.args.iterations}")
            
            # Checkpoint
            if iteration % 5000 == 0 and iteration > 0:
                self.save_checkpoint(iteration)
        
        # Final save
        self.save_checkpoint(self.args.iterations)
        print("Training complete!")


def main():
    args = parse_args()
    trainer = GaussianSplattingTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
