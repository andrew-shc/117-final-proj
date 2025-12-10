"""
Custom 3D Gaussian Splatting Training Script
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from gaussian_model import GSplats
from renderer import Rasterizer
from tqdm import tqdm
import numpy as np
from PIL import Image as PILImage


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iteration = 0

        self.output_path = Path(args.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize Gaussian model
        self.gaussians = GSplats(device=self.device)

        # Rasterizer for rendering
        self.rasterizer = None

        # Ground truth images directory
        self.images_dir = None

        print(f"Device: {self.device}")
        print(f"Output: {self.output_path}")

    def load_colmap_data(self):
        """Load COLMAP sparse reconstruction and initialize Gaussians"""
        # Load COLMAP reconstruction
        self.gaussians.load_from_colmap(
            self.args.data_path,
            sparse_subdir="sparse/0",
            frame_skip=self.args.frame_skip
        )

        # Initialize Gaussians from point cloud
        self.gaussians.init_from_point_cloud(
            initial_scale=0.01,
            initial_opacity=0.5
        )

        # Initialize rasterizer
        self.rasterizer = Rasterizer(self.gaussians)

        # Set up images directory for lazy loading
        self.setup_images_dir()

    def setup_images_dir(self):
        """Set up images directory path and verify it exists"""
        data_path = Path(self.args.data_path)
        self.images_dir = data_path / "images"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found at {self.images_dir}")

        print(f"Images directory: {self.images_dir}")

        # Get image dimensions from first available image
        for image_id, image in self.gaussians.images.items():
            image_path = self.images_dir / image.name
            if image_path.exists():
                pil_img = PILImage.open(image_path)
                original_width, original_height = pil_img.size
                camera = self.gaussians.cameras[image.camera_id]
                print(f"Original image resolution: {original_width}x{original_height}")
                print(f"Camera resolution: {camera.width}x{camera.height}")
                if self.args.resolution_scale != 1.0:
                    scaled_width = int(camera.width * self.args.resolution_scale)
                    scaled_height = int(camera.height * self.args.resolution_scale)
                    print(f"Scaled resolution: {scaled_width}x{scaled_height} (scale factor: {self.args.resolution_scale})")
                break

        # Verify all images exist
        missing_images = []
        for image_id, image in self.gaussians.images.items():
            image_path = self.images_dir / image.name
            if not image_path.exists():
                missing_images.append(image.name)

        if missing_images:
            print(f"Warning: {len(missing_images)} images not found")
            for name in missing_images[:5]:  # Show first 5
                print(f"  - {name}")
            if len(missing_images) > 5:
                print(f"  ... and {len(missing_images) - 5} more")

    def load_gt_image(self, image_id):
        """
        Load a single ground truth image on demand

        Args:
            image_id: ID of the image to load

        Returns:
            img_tensor: torch.Tensor of shape HxWx3, range [0, 1]
        """
        image = self.gaussians.images[image_id]
        camera = self.gaussians.cameras[image.camera_id]
        image_path = self.images_dir / image.name

        # Load image
        pil_img = PILImage.open(image_path)
        
        # Scale image if needed
        if self.args.resolution_scale != 1.0:
            new_width = int(camera.width * self.args.resolution_scale)
            new_height = int(camera.height * self.args.resolution_scale)
            pil_img = pil_img.resize((new_width, new_height), PILImage.LANCZOS)
        
        img_array = np.array(pil_img).astype(np.float32) / 255.0  # HxWx3, range [0, 1]

        # Convert to torch tensor
        img_tensor = torch.from_numpy(img_array).to(self.device)

        return img_tensor
    
    def train_iteration(self, image_id):
        """
        Single training iteration for one image

        Args:
            image_id: ID of the image to render and compute loss

        Returns:
            loss: L1 loss value
        """
        # Get camera and image info
        image = self.gaussians.images[image_id]
        camera = self.gaussians.cameras[image.camera_id]

        # Calculate scaled resolution
        if self.args.resolution_scale != 1.0:
            width = int(camera.width * self.args.resolution_scale)
            height = int(camera.height * self.args.resolution_scale)
        else:
            width = camera.width
            height = camera.height

        # Render the image at scaled resolution (returns PyTorch tensor with gradients)
        rendered_tensor = self.rasterizer.rasterize(camera, image, width=width, height=height, verbose=False)

        # Load ground truth image on demand (lazy loading, automatically scaled)
        gt_tensor = self.load_gt_image(image_id)

        # Compute L1 loss
        loss = torch.abs(rendered_tensor - gt_tensor).mean()

        return loss

    def save_checkpoint(self, iteration, verbose=True):
        """Save checkpoint"""
        checkpoint_path = self.output_path / f"checkpoint_{iteration}.pth"
        self.gaussians.save(checkpoint_path, verbose=verbose)

    def save_ply(self, filename="model.ply", verbose=True):
        """Save model as PLY file"""
        ply_path = self.output_path / filename
        self.gaussians.save_ply(ply_path, verbose=verbose)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        self.gaussians.load(checkpoint_path)

    def render(self, output_subdir="renders", debug=False):
        """Render all camera views"""
        render_dir = self.output_path / output_subdir
        rasterizer = Rasterizer(self.gaussians)
        rasterizer.render_dataset(render_dir, verbose=True, debug=debug)

    def render_test_image(self, iteration, image_id=None):
        """
        Render a single test image during training to track progress
        
        Args:
            iteration: current training iteration
            image_id: specific image to render (if None, uses first image)
        """
        # Use first image if not specified
        if image_id is None:
            for idx, (_, image) in enumerate(self.gaussians.images.items()):
                image = image
                break
            # image_id = list(self.gaussians.images.keys())[0]
        
        # Get camera and image info
        # image = self.gaussians.images[image_id]
        camera = self.gaussians.cameras[image.camera_id]
        
        # Calculate scaled resolution
        if self.args.resolution_scale != 1.0:
            width = int(camera.width * self.args.resolution_scale)
            height = int(camera.height * self.args.resolution_scale)
        else:
            width = camera.width
            height = camera.height
        
        # Render the test image
        with torch.no_grad():  # No gradients needed for visualization
            rendered_tensor = self.rasterizer.rasterize(camera, image, width=width, height=height, verbose=False)
        
        # Convert to numpy and save
        rendered_np = rendered_tensor.detach().cpu().numpy()
        rendered_np = np.clip(rendered_np, 0, 1)
        rendered_uint8 = (rendered_np * 255).astype(np.uint8)
        
        # Save to progress folder
        progress_dir = self.output_path / "training_progress"
        progress_dir.mkdir(exist_ok=True)
        
        output_path = progress_dir / f"iter_{iteration:06d}.png"
        pil_image = PILImage.fromarray(rendered_uint8)
        pil_image.save(output_path)

    def train(self):
        """Main training loop"""
        print("=" * 50)
        print("Starting training")
        print("=" * 50)

        # Load data
        self.load_colmap_data()

        # Get optimizable parameters
        params = self.gaussians.get_optimizable_parameters()

        # Setup optimizer with weight decay for regularization
        optimizer = torch.optim.Adam(params, lr=self.args.learning_rate, eps=1e-15, weight_decay=0.0)

        # Learning rate scheduler - exponential decay
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

        # Get list of image IDs for random sampling
        image_ids = list(self.gaussians.images.keys())

        # Training loop with progress bar
        pbar = tqdm(range(self.args.iterations), desc="Training")
        ema_loss = None
        ema_decay = 0.9

        for iteration in pbar:
            self.iteration = iteration

            # Randomly select an image
            image_id = np.random.choice(image_ids)

            # Zero gradients
            optimizer.zero_grad()

            # Training step - compute loss
            loss = self.train_iteration(image_id)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            # Optimizer step
            optimizer.step()

            # Clamp parameters to valid ranges after optimization
            with torch.no_grad():
                # Clamp opacities to [0, 1]
                if self.gaussians.opacities is not None:
                    self.gaussians.opacities.clamp_(0.0, 1.0)
                # Clamp colors to [0, 1]
                if self.gaussians.colors is not None:
                    self.gaussians.colors.clamp_(0.0, 1.0)
                # Clamp scales to positive values (min 1e-5)
                if self.gaussians.scales is not None:
                    self.gaussians.scales.clamp_(1e-5, 10.0)

            # Update EMA loss for smoother display
            current_loss = loss.item()
            if ema_loss is None:
                ema_loss = current_loss
            else:
                ema_loss = ema_decay * ema_loss + (1 - ema_decay) * current_loss

            # Update progress bar with smoothed loss
            pbar.set_postfix({"loss": f"{ema_loss:.6f}", "raw": f"{current_loss:.6f}"})

            # Learning rate decay
            scheduler.step()

            # Save most recent PLY (overwrites previous, silent)
            self.save_ply("model_latest.ply", verbose=False)

            # Render test image periodically if enabled
            if self.args.save_progress_images and iteration % self.args.progress_image_interval == 0:
                self.render_test_image(iteration)

            # Periodic checkpoint (numbered)
            if iteration % 5000 == 0 and iteration > 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.save_checkpoint(iteration, verbose=True)
                print(f"\nCheckpoint saved at iteration {iteration}, lr: {current_lr:.2e}")

        # Final save
        self.save_checkpoint(self.args.iterations)
        print("\nTraining complete!")


def main():
    # Example usage:
    # Render only: python 3dgs/custom_gaussian_splatter/train.py --data_path ./data/IMG_9184 --render_only
    # Train: python 3dgs/custom_gaussian_splatter/train.py --data_path ./data/IMG_9184 --iterations 10000

    parser = argparse.ArgumentParser(description="Train Custom 3D Gaussian Splatting")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data directory with sparse/0 subdirectory")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Load every Nth frame (1 = all frames, 10 = every 10th frame)")
    parser.add_argument("--resolution_scale", type=float, default=1.0,
                        help="Scale factor for image resolution (e.g., 0.5 for half size, 0.25 for quarter)")
    parser.add_argument("--save_progress_images", action="store_true",
                        help="Save rendered test images during training to track progress")
    parser.add_argument("--progress_image_interval", type=int, default=100,
                        help="Save progress image every N iterations (default: 100)")
    parser.add_argument("--render_only", action="store_true",
                        help="Only render, don't train")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info about camera and Gaussians")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (.pth) to resume from")
    parser.add_argument("--ply_path", type=str, default=None,
                        help="Path to PLY file to load Gaussians from (for rendering)")
    parser.add_argument("--save_ply", action="store_true",
                        help="Save model as PLY file after loading (useful for converting checkpoints to PLY)")

    args = parser.parse_args()
    trainer = Trainer(args)

    if args.render_only:
        # Just load data and render
        if args.ply_path:
            # Load Gaussians from PLY file
            print(f"Loading Gaussians from PLY: {args.ply_path}")
            trainer.gaussians.load_ply(args.ply_path)
            # Still need cameras/images from COLMAP
            trainer.gaussians.load_from_colmap(args.data_path, sparse_subdir="sparse/0", frame_skip=args.frame_skip)
        elif args.checkpoint:
            # Load from checkpoint
            trainer.load_checkpoint(args.checkpoint)
            # Still need cameras/images from COLMAP
            trainer.gaussians.load_from_colmap(args.data_path, sparse_subdir="sparse/0", frame_skip=args.frame_skip)
        else:
            # Load from COLMAP and initialize
            trainer.load_colmap_data()

        # Optionally save as PLY (useful for converting checkpoints)
        if args.save_ply:
            trainer.save_ply("model_converted.ply")

        trainer.render(debug=args.debug)
    else:
        # Normal training
        if args.checkpoint:
            # Resume from checkpoint
            trainer.load_checkpoint(args.checkpoint)
            trainer.gaussians.load_from_colmap(args.data_path, sparse_subdir="sparse/0", frame_skip=args.frame_skip)

        trainer.train()

# python 3dgs/custom_gaussian_splatter/train.py --data_path ./data/IMG_9184 --render_only --frame_skip 10
# python 3dgs/custom_gaussian_splatter/train.py \
#  --data_path ./data/IMG_9184 --iterations 10000 --frame_skip 10 \
#  --resolution_scale 0.25 --learning_rate=0.005 \
#  --save_progress_images --progress_image_interval 1

# python 3dgs/custom_gaussian_splatter/train.py --data_path ./data/IMG_9184 --render_only --ply_path ./outputs/model_latest.ply --frame_skip 10


if __name__ == "__main__":
    main()

