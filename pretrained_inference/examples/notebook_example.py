"""
Jupyter Notebook Example (can also run as a script)

This demonstrates using the models in a notebook-style workflow.
Convert to .ipynb or run as-is.
"""

# Cell 1: Imports
print("Importing libraries...")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pointe_inference import PointEInference
from models.shape_inference import ShapEInference
from utils.visualization import (
    visualize_point_cloud_matplotlib,
    visualize_point_cloud_plotly,
    save_point_cloud_ply
)

# Cell 2: Initialize Point-E
print("\n" + "="*60)
print("Initializing Point-E model...")
print("="*60)

pointe = PointEInference()
pointe.load_models(use_upsampler=True, guidance_scale=3.0)

# Cell 3: Generate with Point-E
print("\n" + "="*60)
print("Generating point cloud with Point-E...")
print("="*60)

prompt = "a vintage red telephone"
point_clouds = pointe.generate_from_text(
    prompt=prompt,
    num_samples=2
)

# Cell 4: Visualize Point-E results
print("\n" + "="*60)
print("Saving Point-E results...")
print("="*60)

output_dir = Path("outputs/notebook/pointe")
output_dir.mkdir(parents=True, exist_ok=True)

for i, pc in enumerate(point_clouds):
    # Save PLY
    save_point_cloud_ply(pc, str(output_dir / f"pointe_sample_{i}.ply"))

    # Save visualization
    visualize_point_cloud_matplotlib(
        pc,
        output_path=str(output_dir / f"pointe_sample_{i}.png"),
        title=f"Point-E: {prompt} (sample {i+1})"
    )

    # Save interactive
    visualize_point_cloud_plotly(
        pc,
        output_path=str(output_dir / f"pointe_sample_{i}.html"),
        title=f"Point-E: {prompt} (sample {i+1})"
    )

print(f"\nPoint-E results saved to: {output_dir}")

# Cell 5: Initialize Shap-E
print("\n" + "="*60)
print("Initializing Shap-E model...")
print("="*60)

shape = ShapEInference()
shape.load_model(model_type='text300M')

# Cell 6: Generate with Shap-E
print("\n" + "="*60)
print("Generating 3D object with Shap-E...")
print("="*60)

prompt = "an avocado armchair"
latents = shape.generate_from_text(
    prompt=prompt,
    num_samples=1,
    guidance_scale=15.0,
    num_steps=64
)

# Cell 7: Save Shap-E results
print("\n" + "="*60)
print("Saving Shap-E results...")
print("="*60)

output_dir = Path("outputs/notebook/shape")
output_dir.mkdir(parents=True, exist_ok=True)

for i, latent in enumerate(latents):
    # Save mesh
    mesh_path = output_dir / f"shape_mesh_{i}.ply"
    shape.latent_to_mesh(latent, str(mesh_path))

    # Save point cloud
    pc = shape.latent_to_point_cloud(latent, num_points=4096)
    pc_path = output_dir / f"shape_pc_{i}.ply"
    save_point_cloud_ply(pc, str(pc_path))

    # Optionally render views (uncomment to use)
    # render_dir = output_dir / f"shape_renders_{i}"
    # shape.render_latent(latent, str(render_dir))

print(f"\nShap-E results saved to: {output_dir}")

# Cell 8: Compare results
print("\n" + "="*60)
print("Comparison Summary")
print("="*60)
print(f"\nPoint-E Results:")
print(f"  Prompt: '{prompt}'")
print(f"  Samples: {len(point_clouds)}")
print(f"  Points per sample: {len(point_clouds[0])}")
print(f"  Location: outputs/notebook/pointe/")

print(f"\nShap-E Results:")
print(f"  Prompt: '{prompt}'")
print(f"  Samples: {len(latents)}")
print(f"  Location: outputs/notebook/shape/")

print("\n" + "="*60)
print("All done! Check the outputs folders to view your results.")
print("="*60)
