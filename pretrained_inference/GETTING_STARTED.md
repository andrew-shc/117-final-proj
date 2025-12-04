# Getting Started - 5 Minute Quick Test

This guide will get you generating 3D point clouds in 5 minutes or less!

## Step 1: Install Dependencies (2-3 minutes)

```bash
# From the project root directory
pip install -r pretrained_inference/requirements.txt
```

This will install PyTorch, Point-E, Shap-E, and visualization libraries.

## Step 2: Run Your First Generation (1-2 minutes)

### Option A: Point-E (Recommended for first test)

```bash
python pretrained_inference/examples/quick_test_pointe.py
```

This will:
1. Download pretrained Point-E model (first time only, ~150MB)
2. Generate a point cloud of "a red motorcycle" (default prompt)
3. Save it to `outputs/pointe/sample_00.ply`
4. Create a visualization image

### Option B: Shap-E (Meshes + Point Clouds)

```bash
python pretrained_inference/examples/quick_test_shape.py
```

This will:
1. Download pretrained Shap-E model (first time only, ~150MB)
2. Generate a mesh of "a shark" (default prompt)
3. Save it to `outputs/shape/sample_00.ply`

## Step 3: View Your Results

### Quick Preview
Check the visualization images in the outputs folder:
- Point-E: `outputs/pointe/sample_00_matplotlib.png`
- Shap-E: Check the PLY files with any 3D viewer

### Interactive Viewing
For interactive 3D viewing, use:

```bash
# Generate with interactive Plotly visualization
python pretrained_inference/examples/quick_test_pointe.py --visualize plotly
```

Then open `outputs/pointe/sample_00_interactive.html` in your browser!

### Professional 3D Viewers
Download and install any of these free tools to view PLY files:
- **MeshLab**: https://www.meshlab.net/ (recommended)
- **CloudCompare**: https://www.cloudcompare.org/
- **Blender**: https://www.blender.org/

## Step 4: Try Your Own Prompts

```bash
# Generate different objects
python pretrained_inference/examples/quick_test_pointe.py --prompt "a blue chair"
python pretrained_inference/examples/quick_test_pointe.py --prompt "a coffee mug"
python pretrained_inference/examples/quick_test_pointe.py --prompt "a wooden table"

# Generate multiple variations
python pretrained_inference/examples/quick_test_pointe.py --prompt "a red car" --samples 3

# Higher quality with more guidance
python pretrained_inference/examples/quick_test_pointe.py --prompt "a palm tree" --guidance-scale 5.0
```

## What's Next?

### Try Different Models
```bash
# Shap-E for better meshes
python pretrained_inference/examples/quick_test_shape.py --prompt "an avocado armchair"

# With rendering from multiple viewpoints
python pretrained_inference/examples/quick_test_shape.py --prompt "a donut" --render
```

### Batch Generation
```bash
# Test multiple prompts at once
python pretrained_inference/examples/batch_generate.py --model pointe
```

### Read the Full Docs
Check out [README.md](README.md) for:
- All command line options
- Tips for better results
- Programmatic API usage
- Troubleshooting guide

## Common Issues

### "CUDA out of memory"
```bash
# Use CPU or reduce quality
python pretrained_inference/examples/quick_test_pointe.py --no-upsampler
```

### "No module named 'point_e'"
```bash
# Reinstall dependencies
pip install -r pretrained_inference/requirements.txt --force-reinstall
```

### Models downloading slowly
This is normal for first run. The models are ~150MB each. Subsequent runs will be much faster as they use cached models.

## Example Prompts to Try

Here are some prompts that work well:

**Simple Objects:**
- "a red apple"
- "a coffee mug"
- "a wooden chair"
- "a laptop computer"

**Vehicles:**
- "a red sports car"
- "a yellow school bus"
- "a motorcycle"
- "a bicycle"

**Animals:**
- "a small cat"
- "a golden dog"
- "a blue bird"
- "a green frog"

**Creative:**
- "an avocado armchair"
- "a donut"
- "a cactus in a pot"
- "a vintage telephone"

**Pro Tip:** More descriptive prompts with colors and materials work better!

---

That's it! You're now generating 3D point clouds with AI. Have fun exploring!
