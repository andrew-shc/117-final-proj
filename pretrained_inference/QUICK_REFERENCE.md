# Quick Reference Card

## Installation
```bash
pip install -r pretrained_inference/requirements.txt
```

## Basic Commands

### Point-E (Text → Point Cloud)
```bash
# Default generation
python pretrained_inference/examples/quick_test_pointe.py

# Custom prompt
python pretrained_inference/examples/quick_test_pointe.py --prompt "YOUR TEXT HERE"

# Multiple samples
python pretrained_inference/examples/quick_test_pointe.py --prompt "a car" --samples 3

# Fast mode (no upsampler)
python pretrained_inference/examples/quick_test_pointe.py --no-upsampler

# With interactive visualization
python pretrained_inference/examples/quick_test_pointe.py --visualize plotly
```

### Shap-E (Text → Mesh + Point Cloud)
```bash
# Default generation
python pretrained_inference/examples/quick_test_shape.py

# Custom prompt
python pretrained_inference/examples/quick_test_shape.py --prompt "YOUR TEXT HERE"

# With multi-view renders
python pretrained_inference/examples/quick_test_shape.py --render

# High quality
python pretrained_inference/examples/quick_test_shape.py --steps 128 --guidance-scale 20
```

### Batch Generation
```bash
# Use default prompts
python pretrained_inference/examples/batch_generate.py --model pointe

# Use custom prompts file
python pretrained_inference/examples/batch_generate.py --model shape --prompts my_prompts.txt
```

## Key Parameters

| Parameter | Point-E | Shap-E | Description |
|-----------|---------|--------|-------------|
| `--prompt` | ✓ | ✓ | Text description |
| `--samples` | ✓ | ✓ | Number to generate |
| `--guidance-scale` | 3-5 | 10-20 | Higher = more faithful |
| `--steps` | - | 64-128 | Diffusion steps |
| `--no-upsampler` | ✓ | - | Faster, lower res |
| `--render` | - | ✓ | Multi-view renders |
| `--output-dir` | ✓ | ✓ | Save location |

## Output Locations
- Point-E: `outputs/pointe/`
- Shap-E: `outputs/shape/`
- Batch: `outputs/batch/`

## File Types
- `.ply` - Point cloud/mesh (open with MeshLab, CloudCompare, Blender)
- `.png` - Matplotlib visualization
- `.html` - Interactive Plotly visualization

## Programmatic Usage
```python
# Point-E (guidance_scale is set during model loading)
from pretrained_inference.models.pointe_inference import PointEInference

model = PointEInference()
model.load_models(use_upsampler=True, guidance_scale=3.0)
pcs = model.generate_from_text("a red car")
model.save_point_cloud(pcs[0], "output.ply")

# Shap-E
from pretrained_inference.models.shape_inference import ShapEInference

model = ShapEInference()
model.load_model()
latents = model.generate_from_text("a shark")
model.latent_to_mesh(latents[0], "output.ply")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Use `--no-upsampler` or `--samples 1` |
| Slow generation | Normal - first run downloads models |
| Import errors | `pip install -r requirements.txt --force-reinstall` |
| Poor quality | Increase `--guidance-scale` or `--steps` |

## Tips
1. Be specific in prompts: "a red sports car" > "car"
2. Include colors and materials for better results
3. Point-E is faster, Shap-E is higher quality
4. First run downloads ~150MB models (one-time)
5. GPU recommended but works on CPU

## Model Comparison

| Feature | Point-E | Shap-E |
|---------|---------|--------|
| Output | Point clouds | Meshes + point clouds |
| Speed | 1-2 min | 2-3 min |
| Points | 1K-4K | Variable |
| Colors | Yes (RGB) | Textured |
| Best for | Quick tests | Final quality |
