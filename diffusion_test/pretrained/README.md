# Pre-trained Models Directory

This directory is for storing pre-trained model checkpoints.

## Getting Started

To use the point cloud diffusion model, you need to either:

### Option 1: Create an Initialized Checkpoint (Quick Start)

Run the checkpoint creation script to create an initialized model:

```bash
cd ..
python create_pretrained_checkpoint.py
```

This will create a `point_cloud_diffusion.pth` checkpoint file that can be used with the inference examples.

**Note**: This creates an *initialized* (not trained) model. The generated point clouds will be random until you train the model.

### Option 2: Train Your Own Model (Recommended)

Train a model from scratch to get high-quality results:

```bash
# Install dependencies first
cd ..
pip install -r requirements.txt

# Run training
cd examples
python train_example.py
```

Training will:
- Generate synthetic training data (spheres, cubes, torus shapes)
- Train for 100 epochs (adjust as needed)
- Save checkpoints every 10 epochs to this directory
- Save the best model as `point_cloud_diffusion.pth`

## Checkpoint Format

Checkpoints are saved in PyTorch format with the following structure:

```python
{
    'epoch': int,                    # Training epoch
    'model_state_dict': OrderedDict, # Model weights
    'optimizer_state_dict': OrderedDict,  # Optimizer state
    'loss': float,                   # Training/validation loss
}
```

## Loading a Checkpoint

```python
import torch
from models import PointCloudDiffusionModel

# Initialize model
model = PointCloudDiffusionModel(
    num_points=1024,
    point_dim=3,
    hidden_dim=128,
    num_timesteps=1000
)

# Load checkpoint
checkpoint = torch.load('point_cloud_diffusion.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use the model
with torch.no_grad():
    samples = model.sample(batch_size=4, device='cpu')
```

## Expected Files

After training, you should see files like:

```
pretrained/
├── README.md (this file)
├── point_cloud_diffusion.pth          # Best model
├── checkpoint_epoch_10.pth            # Checkpoint at epoch 10
├── checkpoint_epoch_20.pth            # Checkpoint at epoch 20
├── ...
├── samples_epoch_10.png               # Generated samples during training
├── samples_epoch_20.png
└── ...
```

## Model Specifications

The default model configuration:

- **Architecture**: PointNet-based Diffusion Model
- **Number of Points**: 1024
- **Point Dimensions**: 3 (XYZ coordinates)
- **Hidden Dimensions**: 128
- **Diffusion Steps**: 1000
- **Total Parameters**: ~500K trainable parameters

## Tips

1. **GPU Training**: Training is much faster on GPU. The script will automatically use CUDA if available.

2. **Training Time**: On a modern GPU, 100 epochs takes approximately 30-60 minutes with 5000 training samples.

3. **Memory**: The default configuration uses ~2GB of GPU memory. Reduce batch size if you encounter OOM errors.

4. **Quality**: The model typically produces reasonable results after 30-50 epochs, with improvements continuing to epoch 100.

## Troubleshooting

**Issue**: No checkpoint file exists
- **Solution**: Run `python create_pretrained_checkpoint.py` or `python examples/train_example.py`

**Issue**: Model generates poor quality point clouds
- **Solution**: Train for more epochs or increase model capacity (hidden_dim)

**Issue**: Out of memory during training
- **Solution**: Reduce BATCH_SIZE in train_example.py
