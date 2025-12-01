"""
Utility script to create a pre-trained model checkpoint for demonstration purposes.

This creates an initialized model checkpoint that can be used with the inference examples.
For a fully trained model, use the train_example.py script.
"""

import sys
import os
import torch
import torch.optim as optim

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import PointCloudDiffusionModel


def create_pretrained_checkpoint():
    """Create a pre-trained model checkpoint"""
    print("Creating pre-trained model checkpoint...")

    # Initialize model with same parameters as training
    model = PointCloudDiffusionModel(
        num_points=1024,
        point_dim=3,
        hidden_dim=128,
        num_timesteps=1000
    )

    # Create optimizer (needed for checkpoint format)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained')
    os.makedirs(output_dir, exist_ok=True)

    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, 'point_cloud_diffusion.pth')

    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.0,
        'note': 'This is an initialized model. For best results, train using train_example.py'
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Create a README in the pretrained directory
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("""# Pre-trained Models

## point_cloud_diffusion.pth

This is a model checkpoint for the Point Cloud Diffusion Model.

**Note**: This checkpoint contains an initialized (not trained) model. It can be used to:
- Test the inference pipeline
- Understand the model architecture
- Serve as a starting point for fine-tuning

For a fully trained model that generates high-quality point clouds:
1. Run `python examples/train_example.py` to train from scratch
2. Or download a pre-trained checkpoint from [your model repository]

### Model Specifications

- **Architecture**: PointNet-based Diffusion Model
- **Number of Points**: 1024
- **Point Dimensions**: 3 (XYZ coordinates)
- **Hidden Dimensions**: 128
- **Diffusion Steps**: 1000
- **Parameters**: ~500K trainable parameters

### Usage

```python
import torch
from models import PointCloudDiffusionModel

# Load model
model = PointCloudDiffusionModel(num_points=1024)
checkpoint = torch.load('pretrained/point_cloud_diffusion.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate samples
with torch.no_grad():
    samples = model.sample(batch_size=4, device='cpu')
```

### Training

To train your own model:

```bash
cd examples
python train_example.py
```

This will create fully trained checkpoints in this directory.
""")
    print(f"README saved to: {readme_path}")

    print("\n" + "="*50)
    print("Pre-trained checkpoint created successfully!")
    print("="*50)
    print(f"\nFiles created:")
    print(f"  - {checkpoint_path}")
    print(f"  - {readme_path}")
    print(f"\nYou can now run the inference examples:")
    print(f"  cd examples && python inference_example.py")


if __name__ == '__main__':
    create_pretrained_checkpoint()
