"""
Point-E Inference Module

Point-E is OpenAI's text-to-3D point cloud diffusion model.
Uses pretrained models from Hugging Face.
"""

import torch
import numpy as np
from typing import Optional, List
from pathlib import Path


class PointEInference:
    """Wrapper for Point-E pretrained model inference."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize Point-E model.

        Args:
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Point-E on {self.device}")

        try:
            from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
            from point_e.diffusion.sampler import PointCloudSampler
            from point_e.models.download import load_checkpoint
            from point_e.models.configs import MODEL_CONFIGS, model_from_config
            from point_e.util.plotting import plot_point_cloud

            self.modules_loaded = True
            self._point_e = {
                'diffusion_from_config': diffusion_from_config,
                'PointCloudSampler': PointCloudSampler,
                'load_checkpoint': load_checkpoint,
                'model_from_config': model_from_config,
                'plot_point_cloud': plot_point_cloud,
                'DIFFUSION_CONFIGS': DIFFUSION_CONFIGS,
                'MODEL_CONFIGS': MODEL_CONFIGS,
            }
        except ImportError as e:
            print(f"Warning: point_e not installed. Install with: pip install point-e")
            self.modules_loaded = False
            return

        self.base_model = None
        self.upsampler_model = None
        self.sampler = None

    def load_models(self, use_upsampler: bool = True, guidance_scale: float = 3.0):
        """
        Load pretrained Point-E models from cache/download.

        Args:
            use_upsampler: Whether to load the upsampler model for higher resolution
            guidance_scale: Classifier-free guidance scale for base model (higher = more faithful to prompt)
        """
        if not self.modules_loaded:
            raise RuntimeError("point_e modules not available")

        print("Loading base model...")
        self.base_model = self._point_e['model_from_config'](
            self._point_e['MODEL_CONFIGS']['base40M-textvec'],
            device=self.device
        )
        self.base_model.eval()

        base_diffusion = self._point_e['diffusion_from_config'](
            self._point_e['DIFFUSION_CONFIGS']['base40M-textvec']
        )

        print("Loading base checkpoint...")
        self.base_model.load_state_dict(
            self._point_e['load_checkpoint']('base40M-textvec', device=self.device)
        )

        if use_upsampler:
            print("Loading upsampler model...")
            self.upsampler_model = self._point_e['model_from_config'](
                self._point_e['MODEL_CONFIGS']['upsample'],
                device=self.device
            )
            self.upsampler_model.eval()

            upsampler_diffusion = self._point_e['diffusion_from_config'](
                self._point_e['DIFFUSION_CONFIGS']['upsample']
            )

            print("Loading upsampler checkpoint...")
            self.upsampler_model.load_state_dict(
                self._point_e['load_checkpoint']('upsample', device=self.device)
            )

            # guidance_scale: [base_model_guidance, upsampler_guidance]
            # Use guidance for base model, disable for upsampler (as per Point-E examples)
            self.sampler = self._point_e['PointCloudSampler'](
                device=self.device,
                models=[self.base_model, self.upsampler_model],
                diffusions=[base_diffusion, upsampler_diffusion],
                num_points=[1024, 4096 - 1024],
                aux_channels=['R', 'G', 'B'],
                guidance_scale=[guidance_scale, 0.0],
                model_kwargs_key_filter=('texts', ''),
                use_karras=[True, True],
                karras_steps=[64, 64],
                sigma_min=[1e-3, 1e-3],
                sigma_max=[120, 160],
                s_churn=[3, 0],
            )
        else:
            self.sampler = self._point_e['PointCloudSampler'](
                device=self.device,
                models=[self.base_model],
                diffusions=[base_diffusion],
                num_points=[1024],
                aux_channels=['R', 'G', 'B'],
                guidance_scale=[guidance_scale],
                model_kwargs_key_filter=('texts',),
                use_karras=[True],
                karras_steps=[64],
                sigma_min=[1e-3],
                sigma_max=[120],
                s_churn=[3],
            )

        print("Models loaded successfully!")

    def generate_from_text(
        self,
        prompt: str,
        num_samples: int = 1
    ) -> List[np.ndarray]:
        """
        Generate 3D point clouds from text prompt.

        Args:
            prompt: Text description of the 3D object
            num_samples: Number of point clouds to generate

        Returns:
            List of point clouds as numpy arrays (N, 6) with XYZ-RGB

        Note:
            Guidance scale is set during model loading, not during generation.
            To change guidance scale, reload the models with a different value.
        """
        if self.sampler is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        print(f"Generating point cloud from prompt: '{prompt}'")
        print("This may take 1-2 minutes on first run (downloading models)...")

        point_clouds = []

        for i in range(num_samples):
            print(f"Generating sample {i+1}/{num_samples}...")

            # Try to use tqdm for progress bar
            try:
                from tqdm.auto import tqdm
                samples = None
                for x in tqdm(
                    self.sampler.sample_batch_progressive(
                        batch_size=1,
                        model_kwargs=dict(texts=[prompt]),
                    ),
                    desc="Diffusion sampling"
                ):
                    samples = x
            except ImportError:
                # Fallback without tqdm
                samples = None
                print("  (sampling in progress, this takes ~1-2 minutes...)")
                for j, x in enumerate(self.sampler.sample_batch_progressive(
                    batch_size=1,
                    model_kwargs=dict(texts=[prompt]),
                )):
                    samples = x
                    # Print periodic updates
                    if j % 10 == 0 and j > 0:
                        print(f"  Step {j}...")

            # Convert to point cloud format
            pc_obj = self.sampler.output_to_point_clouds(samples)[0]

            # Convert PointCloud object to numpy array (N, 6) with XYZ-RGB
            coords = pc_obj.coords  # (N, 3)
            colors = pc_obj.channels['R'][:, None], pc_obj.channels['G'][:, None], pc_obj.channels['B'][:, None]
            colors = np.concatenate(colors, axis=1)  # (N, 3)
            pc = np.concatenate([coords, colors], axis=1)  # (N, 6)

            point_clouds.append(pc)

        print(f"Generated {len(point_clouds)} point cloud(s)")
        return point_clouds

    def save_point_cloud(self, point_cloud: np.ndarray, output_path: str):
        """
        Save point cloud to PLY file.

        Args:
            point_cloud: Point cloud array (N, 6) with XYZ-RGB
            output_path: Path to save PLY file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        coords = point_cloud[:, :3]
        colors = point_cloud[:, 3:]

        with open(output_path, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(coords)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')

            for coord, color in zip(coords, colors):
                r, g, b = (color * 255).astype(np.uint8)
                f.write(f'{coord[0]} {coord[1]} {coord[2]} {r} {g} {b}\n')

        print(f"Saved point cloud to {output_path}")
