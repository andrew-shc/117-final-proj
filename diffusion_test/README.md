# 3D Point Cloud Diffusion Model

A complete implementation of a diffusion model for generating and manipulating 3D point clouds, based on Denoising Diffusion Probabilistic Models (DDPM).

## 📁 Project Structure

```
diffusion_test/
├── models/
│   ├── __init__.py
│   └── point_cloud_diffusion.py    # Main diffusion model implementation
├── utils/
│   ├── __init__.py
│   └── point_cloud_utils.py        # Point cloud utilities and visualization
├── examples/
│   ├── inference_example.py        # Examples using pre-trained models
│   └── train_example.py            # Training from scratch
├── pretrained/                     # Pre-trained model checkpoints
├── requirements.txt
└── README.md
```

## 🌟 Features

- **Complete Diffusion Model**: Full DDPM implementation adapted for 3D point clouds
- **PointNet Backbone**: Efficient point cloud processing using PointNet architecture
- **Multiple Generation Modes**:
  - Generate new point clouds from random noise
  - Interpolate between existing point clouds
  - Denoise corrupted point clouds
- **Synthetic Data Generation**: Built-in utilities for creating training data (spheres, cubes, torus)
- **Visualization Tools**: Easy-to-use visualization functions for point clouds
- **Pre-trained Models**: Example checkpoints for quick testing

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

#### 1. Generate Point Clouds from Noise

```python
import torch
from models import PointCloudDiffusionModel

# Load pre-trained model
model = PointCloudDiffusionModel(num_points=1024)
model.load_state_dict(torch.load('pretrained/point_cloud_diffusion.pth')['model_state_dict'])
model.eval()

# Generate samples
with torch.no_grad():
    samples = model.sample(batch_size=4, device='cpu')
    # samples shape: (4, 1024, 3)
```

#### 2. Run Inference Examples

```bash
cd examples
python inference_example.py
```

This will generate:
- `generated_samples.png` - Random point cloud samples
- `interpolation.png` - Interpolation between different shapes
- `denoising.png` - Denoising demonstration
- `single_sample.png` - High-quality single sample

#### 3. Train Your Own Model

```bash
cd examples
python train_example.py
```

Training features:
- Synthetic dataset generation (5000 training samples)
- Validation monitoring
- Checkpoint saving every 10 epochs
- Progress visualization during training

## 📊 Model Architecture

### Point Cloud Diffusion Model

The model consists of:

1. **Denoising Network (PointNet-based)**:
   - Point-wise feature extraction
   - Time embedding integration
   - MLP-based coordinate prediction

2. **Diffusion Process**:
   - Forward process: Gradually adds Gaussian noise
   - Reverse process: Learns to denoise step-by-step
   - Cosine noise schedule for stable training

3. **Key Components**:
   ```python
   - Input: (B, N, 3) point clouds
   - Hidden dim: 128
   - Time steps: 1000
   - Beta schedule: Cosine
   ```

## 🎯 Use Cases

### 1. Point Cloud Generation

Generate diverse 3D shapes from scratch:

```python
# Generate 10 random point clouds
samples = model.sample(batch_size=10, device='cuda')
```

### 2. Shape Interpolation

Smoothly interpolate between two point clouds:

```python
# Create interpolation
interpolated = model.interpolate(
    point_cloud_1,
    point_cloud_2,
    t=500,          # diffusion timestep
    lambda_=0.5     # interpolation factor
)
```

### 3. Point Cloud Denoising

Clean up noisy point cloud data:

```python
# Add noise
noisy_pc = clean_pc + noise

# Denoise using the model
for t in reversed(range(300)):  # partial denoising
    noisy_pc = model.p_sample(noisy_pc, t)
```

## 📈 Training Details

### Dataset

The training script uses synthetic point clouds:
- **Shapes**: Spheres, Cubes, Torus
- **Variations**: Random scaling and rotation
- **Normalization**: Sphere normalization to [-1, 1]
- **Size**: 5000 training + 500 validation samples

### Hyperparameters

```python
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_POINTS = 1024
NUM_TIMESTEPS = 1000
HIDDEN_DIM = 128
```

### Loss Function

Mean Squared Error (MSE) between predicted and actual noise:
```python
loss = MSE(predicted_noise, actual_noise)
```

## 🔬 Technical Details

### Diffusion Schedule

Uses cosine beta schedule for smoother training:
```python
beta_t = cosine_schedule(t)
alpha_t = 1 - beta_t
alpha_bar_t = cumulative_product(alpha_t)
```

### Forward Process

Adding noise at timestep t:
```python
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
```

### Reverse Process

Denoising at timestep t:
```python
predicted_noise = model(x_t, t)
x_{t-1} = denoise(x_t, predicted_noise, t)
```

## 📝 Code Examples

### Custom Point Cloud Processing

```python
from utils import normalize_point_cloud, visualize_point_cloud

# Load your point cloud (N, 3)
pc = load_your_point_cloud()

# Normalize
pc_normalized = normalize_point_cloud(pc, method='sphere')

# Visualize
visualize_point_cloud(
    pc_normalized,
    title="My Point Cloud",
    save_path="output.png"
)
```

### Batch Generation

```python
# Generate multiple batches
all_samples = []
for i in range(10):
    batch = model.sample(batch_size=8, device='cuda')
    all_samples.append(batch)

# Combine
all_samples = torch.cat(all_samples, dim=0)
# Shape: (80, 1024, 3)
```

## 🎨 Visualization

The project includes comprehensive visualization tools:

```python
from utils import visualize_multiple_point_clouds

# Visualize multiple point clouds in a grid
visualize_multiple_point_clouds(
    point_clouds=[pc1, pc2, pc3, pc4],
    titles=['Shape 1', 'Shape 2', 'Shape 3', 'Shape 4'],
    save_path='comparison.png',
    ncols=2
)
```

## 🔧 Customization

### Change Number of Points

```python
model = PointCloudDiffusionModel(
    num_points=2048,  # Instead of 1024
    point_dim=3,
    hidden_dim=256    # Increase capacity
)
```

### Add Color Information

```python
model = PointCloudDiffusionModel(
    num_points=1024,
    point_dim=6,      # XYZ + RGB
    hidden_dim=128
)
```

### Adjust Diffusion Steps

```python
model = PointCloudDiffusionModel(
    num_timesteps=500,  # Faster generation
    beta_start=1e-4,
    beta_end=0.02
)
```

## 📚 References

- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **PointNet**: [PointNet: Deep Learning on Point Sets](https://arxiv.org/abs/1612.00593)
- **Diffusion Models**: [Understanding Diffusion Models](https://arxiv.org/abs/2208.11970)

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- tqdm

## 💡 Tips

1. **GPU Acceleration**: Use CUDA for faster training and generation
2. **Batch Size**: Adjust based on your GPU memory
3. **Visualization**: Set `show=False` when generating many images
4. **Checkpointing**: Save checkpoints regularly during long training runs
5. **Data Augmentation**: Add random rotations and scaling for more robust models

## 🐛 Troubleshooting

**Issue**: Out of memory during training
- **Solution**: Reduce batch size or number of points

**Issue**: Generated point clouds look noisy
- **Solution**: Train for more epochs or increase model capacity

**Issue**: Slow generation
- **Solution**: Reduce number of diffusion timesteps or use GPU

## 📄 License

This is an educational implementation for learning purposes.

## 🤝 Contributing

Feel free to extend this implementation with:
- Conditional generation (class-guided)
- Attention mechanisms
- Alternative architectures (Transformer-based)
- Real-world datasets (ShapeNet, ModelNet)
- Advanced noise schedules

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Happy Point Cloud Generation! 🎉**
