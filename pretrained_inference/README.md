# Pretrained 3D Point Cloud Diffusion Inference

Quick and easy testing of pretrained 3D point cloud diffusion models for inference only - no training required!

This folder provides ready-to-use scripts for generating 3D point clouds and meshes using state-of-the-art pretrained models from OpenAI:
- **Point-E**: Text-to-3D point cloud generation
- **Shap-E**: Text/image-to-3D with both point clouds and textured meshes

## Quick Start

### Installation

1. Create and activate a virtual environment (**IMPORTANT**):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r pretrained_inference/requirements.txt
```

**Note**:
- First run will download pretrained models automatically (~150MB each)
- **Always activate the venv** before running scripts: `source .venv/bin/activate`
- Generation takes ~2 minutes on CPU (use `--no-upsampler` for faster testing)

### Generate Your First 3D Object

#### Using Point-E (fast, good quality point clouds):
```bash
python pretrained_inference/examples/quick_test_pointe.py --prompt "a red car"
```

#### Using Shap-E (meshes + point clouds):
```bash
python pretrained_inference/examples/quick_test_shape.py --prompt "a shark"
```

Outputs will be saved to `outputs/` directory as PLY files and visualizations.

## Usage Examples

### Point-E Examples

Generate multiple samples:
```bash
python pretrained_inference/examples/quick_test_pointe.py \
    --prompt "a wooden chair" \
    --samples 3 \
    --guidance-scale 5.0
```

Fast generation (skip upsampler):
```bash
python pretrained_inference/examples/quick_test_pointe.py \
    --prompt "a coffee mug" \
    --no-upsampler \
    --visualize both
```

### Shap-E Examples

Generate with mesh and renders:
```bash
python pretrained_inference/examples/quick_test_shape.py \
    --prompt "an avocado armchair" \
    --render
```

High-quality generation:
```bash
python pretrained_inference/examples/quick_test_shape.py \
    --prompt "a donut" \
    --steps 128 \
    --guidance-scale 20.0
```

### Batch Generation

Test multiple prompts at once:
```bash
python pretrained_inference/examples/batch_generate.py --model pointe
```

Use custom prompts from file:
```bash
echo -e "a red car\na blue chair\na green apple" > prompts.txt
python pretrained_inference/examples/batch_generate.py \
    --model shape \
    --prompts prompts.txt
```

## Project Structure

```
pretrained_inference/
├── models/                      # Model inference wrappers
│   ├── pointe_inference.py     # Point-E model
│   └── shape_inference.py      # Shap-E model
├── utils/                       # Utilities
│   └── visualization.py        # Visualization and I/O
├── examples/                    # Example scripts
│   ├── quick_test_pointe.py    # Quick Point-E test
│   ├── quick_test_shape.py     # Quick Shap-E test
│   └── batch_generate.py       # Batch generation
├── outputs/                     # Generated outputs (created automatically)
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Available Models

### Point-E
- **Type**: Text-to-point cloud
- **Resolution**: 1K-4K points (with upsampler)
- **Speed**: ~1-2 minutes per generation
- **Quality**: Good for quick prototyping
- **Colors**: Yes (RGB)

### Shap-E
- **Type**: Text-to-3D (mesh + point cloud)
- **Output**: Textured meshes, point clouds, multi-view renders
- **Speed**: ~2-3 minutes per generation
- **Quality**: Higher quality, more detailed
- **Format**: PLY meshes, can export to OBJ

## Visualization Options

The scripts support multiple visualization methods:

1. **Matplotlib** (static 3D plots): Good for quick previews
2. **Plotly** (interactive HTML): Best for exploring results
3. **Open3D** (live viewer): Interactive 3D viewer
4. **PLY files**: Universal format, open with MeshLab, CloudCompare, etc.

## Command Line Arguments

### Common Arguments
- `--prompt TEXT`: Text description of 3D object
- `--samples N`: Number of samples to generate
- `--guidance-scale FLOAT`: Classifier-free guidance (higher = more faithful)
- `--output-dir PATH`: Where to save outputs

### Point-E Specific
- `--no-upsampler`: Skip upsampler for faster generation (1K points instead of 4K)
- `--visualize {matplotlib,plotly,both,none}`: Visualization method

### Shap-E Specific
- `--steps N`: Number of diffusion steps (more = higher quality)
- `--render`: Render multi-view images
- `--mesh-only`: Skip point cloud extraction

## Tips for Best Results

1. **Prompts**: Be specific and descriptive
   - Good: "a red sports car with black wheels"
   - Bad: "car"

2. **Guidance Scale**:
   - Point-E: 3-5 works well
   - Shap-E: 10-20 for best quality

3. **Speed vs Quality**:
   - Fast: Point-E without upsampler
   - Quality: Shap-E with high steps (128+)

4. **GPU**: Both models work on CPU but are much faster on GPU

## Viewing Generated Files

The scripts save point clouds as PLY files. You can view them with:

- **MeshLab** (free): https://www.meshlab.net/
- **CloudCompare** (free): https://www.cloudcompare.org/
- **Blender** (free): Import PLY files
- **Open3D** (programmatic): Already included in dependencies

## Troubleshooting

### Models downloading slowly
The pretrained checkpoints are large (~150MB each). First run will take time.

### Out of memory
- Reduce batch size (`--samples 1`)
- Use `--no-upsampler` for Point-E
- Close other applications

### CUDA errors
- Update PyTorch: `pip install --upgrade torch`
- Check CUDA compatibility with your GPU

### Import errors
```bash
# Reinstall dependencies
pip install -r pretrained_inference/requirements.txt --force-reinstall
```

## Programmatic Usage

You can also use the models programmatically:

```python
from pretrained_inference.models.pointe_inference import PointEInference
from pretrained_inference.utils.visualization import save_point_cloud_ply

# Initialize and load model (guidance_scale is set during model loading)
model = PointEInference()
model.load_models(use_upsampler=True, guidance_scale=3.0)

# Generate point cloud
point_clouds = model.generate_from_text(
    prompt="a vintage telephone",
    num_samples=1
)

# Save result
save_point_cloud_ply(point_clouds[0], "output.ply")
```

## Next Steps

1. Try different prompts and compare results
2. Experiment with guidance scales and sampling parameters
3. Use generated point clouds in your own projects
4. Combine with other 3D processing tools

## References

- Point-E: https://github.com/openai/point-e
- Shap-E: https://github.com/openai/shap-e
- Papers:
  - Point-E: https://arxiv.org/abs/2212.08751
  - Shap-E: https://arxiv.org/abs/2305.02463

## License

This code is for educational and research purposes. The pretrained models are from OpenAI and subject to their respective licenses.
