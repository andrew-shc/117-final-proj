"""
Gaussian Splatting Model - Manages 3D Gaussian parameters and COLMAP data loading
"""

from typing import Optional, Dict, List, TYPE_CHECKING
import torch
import numpy as np
import pycolmap
from pathlib import Path

if TYPE_CHECKING:
    import pycolmap


class GSplats:
    """
    Gaussian Splats model - stores and manages 3D Gaussian parameters
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize Gaussian Splats model

        Args:
            device: torch device ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Gaussian parameters (initialized as None)
        self.positions: Optional[torch.Tensor] = None      # Nx3 - 3D positions
        self.colors: Optional[torch.Tensor] = None         # Nx3 - RGB colors (0-1 range)
        self.scales: Optional[torch.Tensor] = None         # Nx3 - scale parameters
        self.rotations: Optional[torch.Tensor] = None      # Nx4 - quaternions [w, x, y, z]
        self.opacities: Optional[torch.Tensor] = None      # Nx1 - opacity values

        # COLMAP data
        self.cameras: Optional[Dict[int, 'pycolmap.Camera']] = None
        self.images: Optional[Dict[int, 'pycolmap.Image']] = None
        self.points3D: Optional[Dict[int, 'pycolmap.Point3D']] = None

        print(f"GSplats initialized on device: {self.device}")

    def load_from_colmap(self, data_path: str, sparse_subdir: str = "sparse/0", frame_skip: int = 1) -> 'GSplats':
        """
        Load point cloud from COLMAP reconstruction

        Args:
            data_path: path to data directory
            sparse_subdir: subdirectory containing COLMAP reconstruction
            frame_skip: load every Nth frame (1 = all frames, 2 = every other frame, etc.)

        Returns:
            self: for method chaining
        """
        data_path = Path(data_path)
        sparse_path = data_path / sparse_subdir

        if not sparse_path.exists():
            raise FileNotFoundError(f"COLMAP reconstruction not found at {sparse_path}")

        print(f"Loading COLMAP data from: {sparse_path}")

        # Load reconstruction using pycolmap
        reconstruction = pycolmap.Reconstruction(str(sparse_path))

        # Store cameras
        self.cameras = reconstruction.cameras

        # Filter images based on frame_skip
        all_images = reconstruction.images
        if frame_skip > 1:
            # Sort images by ID to ensure consistent ordering
            sorted_image_ids = sorted(all_images.keys())
            # Select every Nth image
            selected_ids = sorted_image_ids[::frame_skip]
            self.images = {img_id: all_images[img_id] for img_id in selected_ids}
            print(f"Frame skip: {frame_skip} (selected {len(self.images)}/{len(all_images)} images)")
        else:
            self.images = all_images

        self.points3D = reconstruction.points3D

        print(f"Loaded {len(self.cameras)} cameras")
        print(f"Loaded {len(self.images)} images")
        print(f"Loaded {len(self.points3D)} 3D points")

        return self

    def init_from_point_cloud(self, *, initial_scale: float, initial_opacity: float) -> 'GSplats':
        """
        Initialize Gaussian parameters from COLMAP point cloud

        Args:
            initial_scale: initial scale for all Gaussians
            initial_opacity: initial opacity for all Gaussians

        Returns:
            self: for method chaining
        """
        if self.points3D is None or len(self.points3D) == 0:
            raise ValueError("No point cloud data loaded. Call load_from_colmap() first.")

        print("Initializing Gaussians from point cloud...")
        num_points = len(self.points3D)

        # Extract positions and colors from COLMAP points
        xyz = np.zeros((num_points, 3))
        rgb = np.zeros((num_points, 3))

        for idx, (point_id, point) in enumerate(self.points3D.items()):
            xyz[idx] = point.xyz
            rgb[idx] = point.color

        # Convert to torch tensors and move to device
        self.positions = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        self.colors = torch.tensor(rgb / 255.0, dtype=torch.float32, device=self.device)

        # Initialize Gaussian parameters
        # Scales: start with uniform small spheres
        self.scales = torch.ones((num_points, 3), device=self.device) * initial_scale

        # Rotations: start with identity quaternion [w=1, x=0, y=0, z=0]
        self.rotations = torch.zeros((num_points, 4), device=self.device)
        self.rotations[:, 0] = 1.0  # w component

        # Opacities: start semi-transparent
        self.opacities = torch.ones((num_points, 1), device=self.device) * initial_opacity

        print(f"Initialized {num_points} Gaussians")
        print(f"  - Positions: {self.positions.shape}")
        print(f"  - Colors: {self.colors.shape}")
        print(f"  - Scales: {self.scales.shape} (init: {initial_scale})")
        print(f"  - Rotations: {self.rotations.shape} (identity quaternions)")
        print(f"  - Opacities: {self.opacities.shape} (init: {initial_opacity})")

        return self

    @property
    def num_gaussians(self):
        """Return number of Gaussians"""
        if self.positions is None:
            return 0
        return self.positions.shape[0]

    def get_parameters(self):
        """
        Get all Gaussian parameters as a dictionary

        Returns:
            dict: Dictionary containing all parameters
        """
        return {
            'positions': self.positions,
            'colors': self.colors,
            'scales': self.scales,
            'rotations': self.rotations,
            'opacities': self.opacities,
        }

    def get_optimizable_parameters(self):
        """
        Get parameters that should be optimized during training

        Returns:
            list: List of torch tensors with requires_grad=True
        """
        params = []

        if self.positions is not None:
            self.positions.requires_grad = True
            params.append(self.positions)

        if self.colors is not None:
            self.colors.requires_grad = True
            params.append(self.colors)

        if self.scales is not None:
            self.scales.requires_grad = True
            params.append(self.scales)

        if self.rotations is not None:
            self.rotations.requires_grad = True
            params.append(self.rotations)

        if self.opacities is not None:
            self.opacities.requires_grad = True
            params.append(self.opacities)

        return params

    def save(self, path, verbose=True):
        """
        Save Gaussian parameters to file

        Args:
            path: path to save file
            verbose: if True, print status message
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = {
            'positions': self.positions.cpu() if self.positions is not None else None,
            'colors': self.colors.cpu() if self.colors is not None else None,
            'scales': self.scales.cpu() if self.scales is not None else None,
            'rotations': self.rotations.cpu() if self.rotations is not None else None,
            'opacities': self.opacities.cpu() if self.opacities is not None else None,
            'num_gaussians': self.num_gaussians,
        }

        torch.save(state_dict, path)
        if verbose:
            print(f"Saved {self.num_gaussians} Gaussians to {path}")

    def load(self, path):
        """
        Load Gaussian parameters from file

        Args:
            path: path to load file
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        print(f"Loading Gaussians from {path}")
        state_dict = torch.load(path, map_location=self.device)

        self.positions = state_dict['positions'].to(self.device) if state_dict['positions'] is not None else None
        self.colors = state_dict['colors'].to(self.device) if state_dict['colors'] is not None else None
        self.scales = state_dict['scales'].to(self.device) if state_dict['scales'] is not None else None
        self.rotations = state_dict['rotations'].to(self.device) if state_dict['rotations'] is not None else None
        self.opacities = state_dict['opacities'].to(self.device) if state_dict['opacities'] is not None else None

        print(f"Loaded {self.num_gaussians} Gaussians")

        return self

    def load_ply(self, path):
        """
        Load Gaussian parameters from PLY file (3D Gaussian Splatting format)
        
        Note: If the PLY has spherical harmonics (f_dc_*), they will be converted to RGB.
        Otherwise, if it has direct RGB colors, those will be used.

        Args:
            path: path to PLY file
        """
        from plyfile import PlyData
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PLY file not found at {path}")

        print(f"Loading Gaussians from PLY: {path}")
        plydata = PlyData.read(str(path))
        vertex = plydata['vertex']
        N = len(vertex)

        # Load positions (xyz)
        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)
        self.positions = torch.from_numpy(xyz).to(self.device)

        # Load colors - check if spherical harmonics or direct RGB
        if 'f_dc_0' in vertex:
            # Load colors from spherical harmonics DC component (f_dc_0, f_dc_1, f_dc_2)
            # These are in SH space, need to convert to RGB [0, 1]
            sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1).astype(np.float32)
            # Convert from SH to RGB: RGB = 0.5 + SH_C0 * sh_dc, where SH_C0 = 0.28209479177387814
            SH_C0 = 0.28209479177387814
            rgb = 0.5 + SH_C0 * sh_dc
            rgb = np.clip(rgb, 0, 1)
            self.colors = torch.from_numpy(rgb).to(self.device)
        elif 'red' in vertex:
            # Direct RGB colors (0-255 or 0-1 range)
            rgb = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1).astype(np.float32)
            # Normalize to [0, 1] if needed
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            self.colors = torch.from_numpy(rgb).to(self.device)
        else:
            raise ValueError("PLY file must contain either spherical harmonics (f_dc_*) or RGB colors (red, green, blue)")

        # Load opacities
        opacity = vertex['opacity'].astype(np.float32)
        self.opacities = torch.from_numpy(opacity).reshape(-1, 1).to(self.device)

        # Load scales
        scale = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1).astype(np.float32)
        self.scales = torch.from_numpy(scale).to(self.device)

        # Load rotations (quaternions: rot_0, rot_1, rot_2, rot_3)
        # PLY format is typically [w, x, y, z] or [x, y, z, w], we use [w, x, y, z]
        rotation = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1).astype(np.float32)
        # Check if we need to reorder (assume PLY is [w, x, y, z] format like standard 3DGS)
        self.rotations = torch.from_numpy(rotation).to(self.device)

        print(f"Loaded {self.num_gaussians} Gaussians from PLY")
        print(f"  - Positions: {self.positions.shape}")
        print(f"  - Colors: {self.colors.shape}")
        print(f"  - Scales: {self.scales.shape}")
        print(f"  - Rotations: {self.rotations.shape}")
        print(f"  - Opacities: {self.opacities.shape}")

        return self

    def save_ply(self, path, verbose=True):
        """
        Save Gaussian parameters to PLY file (simple RGB format, no spherical harmonics)

        Args:
            path: path to save PLY file
            verbose: if True, print status messages
        """
        from plyfile import PlyData, PlyElement
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Saving {self.num_gaussians} Gaussians to PLY: {path}")

        # Convert tensors to numpy
        positions = self.positions.detach().cpu().numpy()
        colors = self.colors.detach().cpu().numpy()
        scales = self.scales.detach().cpu().numpy()
        rotations = self.rotations.detach().cpu().numpy()
        opacities = self.opacities.detach().cpu().numpy().flatten()

        # Clamp colors to valid range
        colors = np.clip(colors, 0, 1)

        # Create structured array for PLY
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),                      # positions
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4'),            # RGB colors (0-1)
            ('opacity', 'f4'),                                          # opacity
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),  # scales
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),  # quaternions [w,x,y,z]
        ]

        elements = np.empty(self.num_gaussians, dtype=dtype)
        
        # Fill in data
        elements['x'] = positions[:, 0]
        elements['y'] = positions[:, 1]
        elements['z'] = positions[:, 2]
        
        elements['red'] = colors[:, 0]
        elements['green'] = colors[:, 1]
        elements['blue'] = colors[:, 2]
        
        elements['opacity'] = opacities
        
        elements['scale_0'] = scales[:, 0]
        elements['scale_1'] = scales[:, 1]
        elements['scale_2'] = scales[:, 2]
        
        elements['rot_0'] = rotations[:, 0]
        elements['rot_1'] = rotations[:, 1]
        elements['rot_2'] = rotations[:, 2]
        elements['rot_3'] = rotations[:, 3]

        # Create PLY element and save
        vertex_element = PlyElement.describe(elements, 'vertex')
        ply_data = PlyData([vertex_element])
        ply_data.write(str(path))

        if verbose:
            print(f"✓ Saved PLY file to {path}")

        return self

    def to(self, device):
        """
        Move all parameters to a different device

        Args:
            device: target device
        """
        self.device = torch.device(device)

        if self.positions is not None:
            self.positions = self.positions.to(self.device)
        if self.colors is not None:
            self.colors = self.colors.to(self.device)
        if self.scales is not None:
            self.scales = self.scales.to(self.device)
        if self.rotations is not None:
            self.rotations = self.rotations.to(self.device)
        if self.opacities is not None:
            self.opacities = self.opacities.to(self.device)

        print(f"Moved Gaussians to {self.device}")

        return self

    def __repr__(self):
        return f"GSplats(num_gaussians={self.num_gaussians}, device={self.device})"

    def densify_and_split(self, avg_gradients: torch.Tensor, gradient_threshold: float, size_threshold: float, N: int = 2):
        """
        Densify Gaussians by splitting large Gaussians with high gradients.
        
        Split strategy:
        - For each Gaussian to split, create N new Gaussians
        - New Gaussians are positioned using samples from the original Gaussian's distribution
        - Scales are divided by (0.8 * N) to maintain similar volume coverage
        
        Args:
            avg_gradients: average gradient norm per Gaussian [num_gaussians, 1]
            gradient_threshold: split Gaussians with avg gradient norm above this
            size_threshold: split Gaussians with max scale above this
            N: number of new Gaussians to create from each split (default: 2)
        """
        if self.positions is None or avg_gradients is None:
            return
        
        # Find Gaussians that are large AND have high gradients
        max_scales = torch.max(self.scales, dim=1)[0]
        high_gradient_mask = (avg_gradients.squeeze() > gradient_threshold)
        large_mask = (max_scales > size_threshold)
        split_mask = high_gradient_mask & large_mask
        
        num_to_split = split_mask.sum().item()
        if num_to_split == 0:
            return
        
        # Get Gaussians to split
        split_positions = self.positions[split_mask]
        split_colors = self.colors[split_mask]
        split_scales = self.scales[split_mask]
        split_rotations = self.rotations[split_mask]
        split_opacities = self.opacities[split_mask]
        
        # Create N new Gaussians for each split
        new_gaussians_list = []
        for i in range(N):
            # Sample positions from the Gaussian distribution
            # Use the scale to determine sampling range
            std = split_scales * 0.5  # Sample within ~half the scale
            samples = torch.randn_like(split_positions) * std
            new_positions = split_positions + samples
            
            # Reduce scale of new Gaussians to maintain coverage
            new_scales = split_scales / (0.8 * N)
            
            new_gaussians_list.append({
                'positions': new_positions,
                'colors': split_colors.clone(),
                'scales': new_scales,
                'rotations': split_rotations.clone(),
                'opacities': split_opacities.clone(),
            })
        
        # Remove original Gaussians that were split
        keep_mask = ~split_mask
        self.positions = self.positions[keep_mask]
        self.colors = self.colors[keep_mask]
        self.scales = self.scales[keep_mask]
        self.rotations = self.rotations[keep_mask]
        self.opacities = self.opacities[keep_mask]
        
        # Add all new Gaussians
        for new_gaussians in new_gaussians_list:
            self.positions = torch.cat([self.positions, new_gaussians['positions']], dim=0)
            self.colors = torch.cat([self.colors, new_gaussians['colors']], dim=0)
            self.scales = torch.cat([self.scales, new_gaussians['scales']], dim=0)
            self.rotations = torch.cat([self.rotations, new_gaussians['rotations']], dim=0)
            self.opacities = torch.cat([self.opacities, new_gaussians['opacities']], dim=0)
        
        print(f"Split {num_to_split} large Gaussians into {num_to_split * N} new Gaussians (total: {self.num_gaussians})")
    
    def densify_and_clone(self, avg_gradients: torch.Tensor, gradient_threshold: float, size_threshold: float):
        """
        Densify Gaussians by cloning small Gaussians with high gradients.
        
        Clone strategy:
        - Duplicate small Gaussians that have high gradients
        - New Gaussians are offset slightly in the direction of the gradient
        - All other parameters are copied exactly
        
        Args:
            avg_gradients: average gradient norm per Gaussian [num_gaussians, 1]
            gradient_threshold: clone Gaussians with avg gradient norm above this
            size_threshold: clone Gaussians with max scale below this
        """
        if self.positions is None or avg_gradients is None:
            return
        
        # Find Gaussians that are small AND have high gradients
        max_scales = torch.max(self.scales, dim=1)[0]
        high_gradient_mask = (avg_gradients.squeeze() > gradient_threshold)
        small_mask = (max_scales <= size_threshold)
        clone_mask = high_gradient_mask & small_mask
        
        num_to_clone = clone_mask.sum().item()
        if num_to_clone == 0:
            return
        
        # Get Gaussians to clone
        clone_positions = self.positions[clone_mask]
        clone_colors = self.colors[clone_mask]
        clone_scales = self.scales[clone_mask]
        clone_rotations = self.rotations[clone_mask]
        clone_opacities = self.opacities[clone_mask]
        
        # Clone with slight offset for better coverage
        # Offset in a random direction by a small fraction of the scale
        offset = torch.randn_like(clone_positions) * clone_scales.mean(dim=1, keepdim=True) * 0.1
        new_positions = clone_positions + offset
        
        # Concatenate cloned Gaussians
        self.positions = torch.cat([self.positions, new_positions], dim=0)
        self.colors = torch.cat([self.colors, clone_colors], dim=0)
        self.scales = torch.cat([self.scales, clone_scales], dim=0)
        self.rotations = torch.cat([self.rotations, clone_rotations], dim=0)
        self.opacities = torch.cat([self.opacities, clone_opacities], dim=0)
        
        print(f"Cloned {num_to_clone} small Gaussians (total: {self.num_gaussians})")
    
    def prune_gaussians(self, opacity_threshold: float = 0.01, extent_threshold: float = None):
        """
        Remove Gaussians with very low opacity or that are too large.
        
        Args:
            opacity_threshold: remove Gaussians with opacity below this
            extent_threshold: remove Gaussians with scale above this (optional)
        """
        if self.positions is None or self.opacities is None:
            return
        
        # Create mask for Gaussians to keep
        keep_mask = self.opacities.squeeze() > opacity_threshold
        
        if extent_threshold is not None and self.scales is not None:
            max_scales = torch.max(self.scales, dim=1)[0]
            keep_mask = keep_mask & (max_scales < extent_threshold)
        
        num_removed = (~keep_mask).sum().item()
        if num_removed == 0:
            return
        
        # Apply mask to all parameters
        self.positions = self.positions[keep_mask]
        self.colors = self.colors[keep_mask]
        self.scales = self.scales[keep_mask]
        self.rotations = self.rotations[keep_mask]
        self.opacities = self.opacities[keep_mask]
        
        print(f"Pruned {num_removed} Gaussians (kept {self.num_gaussians})")
