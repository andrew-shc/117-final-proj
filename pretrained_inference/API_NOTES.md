# Point-E API Notes

## Important: Guidance Scale Parameter

The Point-E API requires the `guidance_scale` parameter to be set during **sampler initialization**, not during sample generation.

### Correct Usage

```python
from pretrained_inference.models.pointe_inference import PointEInference

# Initialize model
model = PointEInference()

# Load models with guidance_scale parameter
model.load_models(
    use_upsampler=True,
    guidance_scale=3.0  # Set guidance scale HERE
)

# Generate (no guidance_scale parameter)
point_clouds = model.generate_from_text(
    prompt="a red car",
    num_samples=1
)
```

### Why This Design?

Point-E uses a multi-stage diffusion pipeline:
1. **Base model** (1024 points) - uses guidance
2. **Upsampler model** (additional 3072 points) - typically no guidance

The `guidance_scale` is passed as a list `[base_guidance, upsampler_guidance]` to the sampler constructor:
- With upsampler: `guidance_scale=[3.0, 0.0]`
- Without upsampler: `guidance_scale=[3.0]`

This is set once when creating the sampler and applies to all subsequent generations.

### Changing Guidance Scale

If you need to change the guidance scale, you must reload the models:

```python
# Generate with guidance_scale=3.0
model.load_models(guidance_scale=3.0)
pcs1 = model.generate_from_text("a car")

# Generate with different guidance_scale=5.0
model.load_models(guidance_scale=5.0)
pcs2 = model.generate_from_text("a car")
```

### Command Line Usage

The example scripts handle this correctly:

```bash
# Default guidance_scale=3.0
python pretrained_inference/examples/quick_test_pointe.py --prompt "a car"

# Custom guidance_scale=5.0
python pretrained_inference/examples/quick_test_pointe.py --prompt "a car" --guidance-scale 5.0
```

## Point Cloud Format

Point-E outputs are converted using `sampler.output_to_point_clouds()` which returns a structured point cloud object with:
- `coords`: XYZ positions (N, 3)
- `channels`: RGB colors (N, 3)

Our wrapper combines these into a single array (N, 6) for easier handling.

## References

- Point-E GitHub: https://github.com/openai/point-e
- Example notebook: https://github.com/openai/point-e/blob/main/point_e/examples/text2pointcloud.ipynb
- Sampler implementation: https://github.com/openai/point-e/blob/main/point_e/diffusion/sampler.py
