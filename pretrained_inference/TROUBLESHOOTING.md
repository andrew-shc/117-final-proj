# Troubleshooting Guide

## Successfully Resolved Issues

### Issue 1: TypeError - guidance_scale parameter
**Error**: `TypeError: PointCloudSampler.sample_batch_progressive() got an unexpected keyword argument 'guidance_scale'`

**Cause**: Point-E requires `guidance_scale` to be set during sampler initialization, not during sampling.

**Solution**: Pass `guidance_scale` to `load_models()` instead of `generate_from_text()`.

**Fixed in**: [models/pointe_inference.py](models/pointe_inference.py)

###Issue 2: TypeError - PointCloud object not subscriptable
**Error**: `TypeError: 'PointCloud' object is not subscriptable`

**Cause**: Point-E's `output_to_point_clouds()` returns a `PointCloud` object with `.coords` and `.channels` attributes, not a numpy array.

**Solution**: Extract coordinates and RGB channels from the PointCloud object and concatenate them into a numpy array.

**Fixed in**: [models/pointe_inference.py](models/pointe_inference.py:176-185)

### Issue 3: AssertionError in PointCloudSampler initialization
**Error**: `AssertionError` when creating sampler with single model

**Cause**: Missing Karras sampling parameters (`use_karras`, `karras_steps`, etc.) which must be sequences matching the number of models.

**Solution**: Added all required Karras parameters as sequences to sampler initialization.

**Fixed in**: [models/pointe_inference.py](models/pointe_inference.py:107-122)

### Issue 4: Long generation time (2.8 hours stuck)
**Cause**: Multiple factors:
- Running on CPU instead of GPU
- Using wrong Python environment
- No progress feedback

**Solutions**:
1. Use `--no-upsampler` flag for faster generation (1 stage instead of 2)
2. Activate the virtual environment: `source .venv/bin/activate`
3. Added tqdm progress bars to show sampling progress
4. GPU would be much faster if available

**Typical timing on CPU**:
- Without upsampler: ~2 minutes (64 iterations)
- With upsampler: ~4-5 minutes (128 iterations)

## Current Known Issues

### CUDA Warnings
**Warning**: `CUDA unknown error - this may be due to an incorrectly set up environment`

**Impact**: Forces CPU usage, which is slower but functional

**Workaround**: Use CPU mode (automatic fallback) or fix CUDA environment if GPU is available

## Best Practices

### 1. Always use the virtual environment
```bash
source .venv/bin/activate
python pretrained_inference/examples/quick_test_pointe.py
```

### 2. Start with `--no-upsampler` for testing
```bash
python pretrained_inference/examples/quick_test_pointe.py \
    --prompt "a red car" \
    --no-upsampler  # 2x faster
```

### 3. Monitor progress
The scripts now show progress bars:
```
Diffusion sampling: 64it [01:57, 1.81s/it]
```

### 4. Expected generation times

**CPU (no GPU)**:
- No upsampler: ~2 minutes
- With upsampler: ~4-5 minutes

**GPU** (if available):
- No upsampler: ~30-60 seconds
- With upsampler: ~1-2 minutes

## Quick Fixes

### "point_e not installed"
```bash
source .venv/bin/activate
pip install git+https://github.com/openai/point-e.git
```

### "Out of memory"
```bash
# Use no-upsampler mode
python pretrained_inference/examples/quick_test_pointe.py --no-upsampler
```

### "Taking too long"
- **Normal on first run**: Downloads models (~150MB each)
- **Normal on CPU**: 2-5 minutes per generation
- **Tip**: Use `--no-upsampler` to cut time in half

### Check if it's working
Look for the progress bar:
```
Diffusion sampling: 42it [01:11, 1.75s/it]
```

If you see iterations counting up, it's working!

## Getting Help

If you encounter issues not covered here:

1. Check that you're using the virtual environment
2. Verify Point-E is installed: `pip show point-e`
3. Look at the error message carefully
4. Try with `--no-upsampler` first
5. Check available disk space (models are ~300MB)

## Performance Tips

### Speed up generation:
1. Use GPU if available (10x faster)
2. Use `--no-upsampler` (2x faster, 1K points instead of 4K)
3. Reduce samples: `--samples 1`
4. Close other CPU-intensive applications

### Improve quality:
1. Use full upsampler (default)
2. Increase guidance: `--guidance-scale 5.0`
3. Be specific in prompts: "a shiny red sports car" vs "a car"

## Success Indicators

You know it's working when you see:
1. ✓ "Models loaded successfully!"
2. ✓ Progress bar: "Diffusion sampling: X/64"
3. ✓ "Generated 1 point cloud(s)"
4. ✓ Files in `outputs/pointe/`:
   - `sample_00.ply` - Point cloud file
   - `sample_00_matplotlib.png` - Visualization

## File Outputs

After successful generation:
```
outputs/pointe/
├── sample_00.ply              # Point cloud (open with MeshLab, CloudCompare)
└── sample_00_matplotlib.png   # Static visualization
```

Open `.ply` files with:
- **MeshLab**: https://www.meshlab.net/
- **CloudCompare**: https://www.cloudcompare.org/
- **Blender**: https://www.blender.org/
