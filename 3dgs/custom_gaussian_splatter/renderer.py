"""
Gaussian Splatting Rasterizer - Class-based renderer for Gaussian splats
Differentiable PyTorch implementation
"""

from typing import Tuple, Dict, Optional, TYPE_CHECKING, Any, Union
import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation
import time

if TYPE_CHECKING:
    from gaussian_model import GSplats
    import pycolmap


# ============================================================================
# Helper Functions
# ============================================================================

def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices (differentiable PyTorch)
    
    Args:
        quaternions: Nx4 quaternions [w, x, y, z] (scalar-first convention)
    
    Returns:
        R: Nx3x3 rotation matrices
    """
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Normalize quaternions
    norm = torch.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Build rotation matrix
    R = torch.zeros((quaternions.shape[0], 3, 3), dtype=quaternions.dtype, device=quaternions.device)
    
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    
    return R


def build_covariance_3d(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """
    Build 3D covariance matrices from scales and rotations (differentiable PyTorch)

    Covariance formula: Σ = R S S^T R^T
    where R is rotation matrix and S is diagonal scale matrix

    Args:
        scales: Nx3 scale parameters (torch.Tensor)
        rotations: Nx4 quaternions [w, x, y, z] (scalar-first convention, torch.Tensor)

    Returns:
        cov: Nx3x3 covariance matrices (torch.Tensor)
    """
    # Convert quaternions to rotation matrices
    R = quaternion_to_rotation_matrix(rotations)  # Nx3x3

    # Build scale matrices (diagonal)
    S = torch.zeros((scales.shape[0], 3, 3), dtype=scales.dtype, device=scales.device)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]

    # Covariance = R @ S @ S^T @ R^T
    RS = torch.bmm(R, S)
    cov = torch.bmm(RS, RS.transpose(1, 2))

    return cov


# ============================================================================
# Rasterizer Class
# ============================================================================

class Rasterizer:
    """
    Gaussian Splatting Rasterizer - Renders Gaussians to images
    """

    def __init__(self, gaussians: 'GSplats', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize rasterizer with Gaussian model

        Args:
            gaussians: GSplats object containing Gaussian parameters (as PyTorch tensors)
            device: device to run computations on ('cuda' or 'cpu')
        """
        self.gaussians = gaussians
        self.device = device

    def get_intrinsics(self, camera: 'pycolmap.Camera', scale: float = 1.0) -> np.ndarray:
        """
        Get camera intrinsics as 3x3 matrix, optionally scaled

        Args:
            camera: pycolmap Camera object
            scale: scale factor for intrinsics (e.g., 0.5 for half resolution)

        Returns:
            K: 3x3 intrinsic matrix
        """
        fx, fy, cx, cy = camera.params[:4]
        
        # Scale intrinsics proportionally
        fx_scaled = fx * scale
        fy_scaled = fy * scale
        cx_scaled = cx * scale
        cy_scaled = cy * scale

        K = np.array([
            [fx_scaled, 0, cx_scaled],
            [0, fy_scaled, cy_scaled],
            [0, 0, 1]
        ])

        return K

    def project_gaussians_to_2d(
        self,
        means_3d: torch.Tensor,
        cov_3d: torch.Tensor,
        camera: 'pycolmap.Camera',
        image: 'pycolmap.Image',
        scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project 3D Gaussians to 2D screen space (differentiable PyTorch)

        Args:
            means_3d: Nx3 3D positions (torch.Tensor)
            cov_3d: Nx3x3 3D covariance matrices (torch.Tensor)
            camera: pycolmap Camera object
            image: pycolmap Image object
            scale: Resolution scale factor for intrinsics (default 1.0)

        Returns:
            means_2d: Nx2 2D screen positions (torch.Tensor)
            cov_2d: Nx2x2 2D covariance matrices (torch.Tensor)
            depths: N depth values (torch.Tensor)
            in_frustum: N boolean mask for valid Gaussians (torch.Tensor)
        """
        # Get camera pose (world-to-camera transform)
        cam_from_world = image.cam_from_world()
        R_cam = torch.from_numpy(cam_from_world.rotation.matrix()).float().to(means_3d.device)  # 3x3
        t_cam = torch.from_numpy(cam_from_world.translation).float().to(means_3d.device)  # 3

        # Transform points to camera space
        means_cam = torch.mm(means_3d, R_cam.T) + t_cam  # Nx3
        depths = means_cam[:, 2]

        # Filter points behind camera
        in_frustum = depths > 0.1

        # Get intrinsics with scale factor
        K_np = self.get_intrinsics(camera, scale)
        K = torch.from_numpy(K_np).float().to(means_3d.device)
        fx, fy = K[0, 0], K[1, 1]

        # Project means to 2D
        means_2d_homo = torch.mm(means_cam, K.T)  # Nx3
        means_2d = means_2d_homo[:, :2] / means_2d_homo[:, 2:3]

        # Check bounds
        in_frustum = in_frustum & (means_2d[:, 0] >= 0) & (means_2d[:, 0] < camera.width)
        in_frustum = in_frustum & (means_2d[:, 1] >= 0) & (means_2d[:, 1] < camera.height)

        # Project covariance to 2D
        # Jacobian of perspective projection
        N = means_3d.shape[0]
        J = torch.zeros((N, 2, 3), dtype=means_3d.dtype, device=means_3d.device)
        z = means_cam[:, 2]
        J[:, 0, 0] = fx / z
        J[:, 0, 2] = -fx * means_cam[:, 0] / (z ** 2)
        J[:, 1, 1] = fy / z
        J[:, 1, 2] = -fy * means_cam[:, 1] / (z ** 2)

        # Transform covariance to camera space
        # For each Gaussian: cov_cam_i = R_cam @ cov_3d_i @ R_cam^T
        cov_cam = torch.einsum('ij,njk,lk->nil', R_cam, cov_3d, R_cam)  # Nx3x3

        # Project to 2D: cov_2d = J @ cov_cam @ J^T
        cov_2d = torch.bmm(torch.bmm(J, cov_cam), J.transpose(1, 2))  # Nx2x2

        # Add small regularization for numerical stability
        cov_2d[:, 0, 0] += 1e-4
        cov_2d[:, 1, 1] += 1e-4

        return means_2d, cov_2d, depths, in_frustum

    def rasterize(
        self,
        camera: 'pycolmap.Camera',
        image: 'pycolmap.Image',
        width: Optional[int] = None,
        height: Optional[int] = None,
        verbose: bool = False,
        debug: bool = False
    ) -> torch.Tensor:
        """
        Render Gaussians for a single camera pose (differentiable PyTorch version)

        Args:
            camera: pycolmap Camera object
            image: pycolmap Image object (contains camera pose)
            width: output width (default: camera width)
            height: output height (default: camera height)
            verbose: if True, print rendering statistics
            debug: if True, print detailed debug info about camera and Gaussians

        Returns:
            rendered: HxWx3 RGB image (0-1 range) as torch.Tensor with gradients
        """
        if width is None:
            width = camera.width
        if height is None:
            height = camera.height

        # Get Gaussian parameters (assumed to be PyTorch tensors)
        positions = self.gaussians.positions
        colors = self.gaussians.colors
        scales = self.gaussians.scales
        rotations = self.gaussians.rotations
        opacities = self.gaussians.opacities

        # Ensure all parameters are on the correct device
        positions = positions.to(self.device)
        colors = colors.to(self.device)
        scales = scales.to(self.device)
        rotations = rotations.to(self.device)
        opacities = opacities.to(self.device)

        total_gaussians = len(positions)

        if debug:
            print(f"\n=== Debug Info ===")
            print(f"Camera intrinsics: fx={camera.params[0]:.1f}, fy={camera.params[1]:.1f}, cx={camera.params[2]:.1f}, cy={camera.params[3]:.1f}")
            print(f"Image resolution: {width}x{height}")
            print(f"Gaussian positions range: [{positions.min().item():.2f}, {positions.max().item():.2f}]")
            print(f"Gaussian scales mean: {scales.mean().item():.4f}, std: {scales.std().item():.4f}")
            print(f"Gaussian opacities mean: {opacities.mean().item():.4f}")
            cam_from_world = image.cam_from_world()
            print(f"Camera position (world): {-cam_from_world.rotation.matrix().T @ cam_from_world.translation}")
            print(f"Camera looking direction: {cam_from_world.rotation.matrix()[2, :]}")
            print(f"==================\n")

        # Build 3D covariance matrices
        cov_3d = build_covariance_3d(scales, rotations)

        # Compute resolution scale factor
        scale_factor = width / camera.width

        # Project to 2D with scaled intrinsics
        means_2d, cov_2d, depths, in_frustum = self.project_gaussians_to_2d(
            positions, cov_3d, camera, image, scale_factor
        )

        # Filter valid Gaussians
        valid_indices = torch.where(in_frustum)[0]
        num_visible = len(valid_indices)

        if verbose:
            print(f"Rendering: {num_visible}/{total_gaussians} Gaussians visible in frustum ({100*num_visible/total_gaussians:.1f}%)")

        if len(valid_indices) == 0:
            return torch.zeros((height, width, 3), dtype=torch.float32, device=self.device)

        means_2d = means_2d[valid_indices]
        cov_2d = cov_2d[valid_indices]
        colors_valid = colors[valid_indices]
        opacities_valid = opacities[valid_indices].flatten()
        depths_valid = depths[valid_indices]

        # Sort by depth (back to front for alpha blending)
        depth_order = torch.argsort(-depths_valid)
        means_2d = means_2d[depth_order]
        cov_2d = cov_2d[depth_order]
        colors_valid = colors_valid[depth_order]
        opacities_valid = opacities_valid[depth_order]

        # Rasterize Gaussians
        rendered = torch.zeros((height, width, 3), dtype=torch.float32, device=self.device)
        alpha_accumulated = torch.zeros((height, width), dtype=torch.float32, device=self.device)

        # Create pixel grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=self.device),
            torch.arange(width, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1)  # HxWx2

        # Render each Gaussian
        for i in range(len(means_2d)):
            # Get Gaussian parameters
            mean = means_2d[i]  # 2D center
            cov = cov_2d[i]  # 2x2 covariance
            color = colors_valid[i]
            opacity = opacities_valid[i]

            # Compute inverse covariance for Gaussian evaluation
            try:
                cov_inv = torch.inverse(cov)
            except RuntimeError:
                continue  # Skip singular matrices

            # Determine bounding box (3 sigma rule)
            eigenvalues = torch.linalg.eigvalsh(cov)
            if torch.any(eigenvalues <= 0):
                continue
            radius = 3.0 * torch.sqrt(torch.max(eigenvalues))

            x_min = max(0, int((mean[0] - radius).item()))
            x_max = min(width, int((mean[0] + radius).item()) + 1)
            y_min = max(0, int((mean[1] - radius).item()))
            y_max = min(height, int((mean[1] + radius).item()) + 1)

            if x_min >= x_max or y_min >= y_max:
                continue

            # Get pixels in bounding box
            pixels = pixel_coords[y_min:y_max, x_min:x_max]  # HxWx2

            # Compute offset from Gaussian center
            delta = pixels - mean  # HxWx2

            # Evaluate Gaussian: exp(-0.5 * delta^T @ cov_inv @ delta)
            # Mahalanobis distance: d^2 = delta^T @ cov_inv @ delta
            delta_flat = delta.reshape(-1, 2)  # (H*W)x2
            mahalanobis = torch.sum(torch.mm(delta_flat, cov_inv) * delta_flat, dim=1)  # H*W
            mahalanobis = mahalanobis.reshape(delta.shape[0], delta.shape[1])  # HxW

            # Gaussian weight
            gaussian_weight = torch.exp(-0.5 * mahalanobis)

            # Alpha blending weight
            alpha = opacity * gaussian_weight

            # Alpha blending (back-to-front)
            transmittance = 1.0 - alpha_accumulated[y_min:y_max, x_min:x_max]
            alpha_blend = alpha * transmittance

            # Accumulate color
            for c in range(3):
                rendered[y_min:y_max, x_min:x_max, c] += alpha_blend * color[c]

            # Update alpha accumulation
            alpha_accumulated[y_min:y_max, x_min:x_max] += alpha_blend

            # Early stopping if fully opaque
            if i % 1000 == 0 and torch.mean(alpha_accumulated) > 0.99:
                break

        # Clamp to valid range
        rendered = torch.clamp(rendered, 0, 1)

        return rendered

    def render_dataset(self, output_dir: str, verbose: bool = True, debug: bool = False) -> Dict[str, np.ndarray]:
        """
        Render all views in the dataset

        Args:
            output_dir: directory to save rendered images
            verbose: if True, print progress and statistics

        Returns:
            rendered_images: dict of {image_name: rendered_array}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.gaussians.cameras is None or self.gaussians.images is None:
            raise ValueError("No camera/image data loaded in gaussians model")

        rendered_images = {}
        num_images = len(self.gaussians.images)

        if verbose:
            print(f"Rendering {num_images} views...")
            print(f"Total Gaussians in scene: {self.gaussians.num_gaussians}")
            print()

        for idx, (_, image) in enumerate(self.gaussians.images.items()):
            camera = self.gaussians.cameras[image.camera_id]

            if verbose:
                print(f"[{idx+1}/{num_images}] {image.name}")

            # Render with statistics
            start_time = time.perf_counter()
            rendered_tensor = self.rasterize(camera, image, verbose=verbose, debug=debug)
            render_time = time.perf_counter() - start_time

            # Convert to numpy for saving
            rendered = rendered_tensor.detach().cpu().numpy()

            # Store in dict
            rendered_images[image.name] = rendered

            # Save to disk
            output_path = output_dir / f"{image.name}.png"
            self.save_image(rendered, output_path)

            if verbose:
                print(f"Render time: {render_time:.3f}s")
                print()

        if verbose:
            print(f"✓ All rendered images saved to {output_dir}")

        return rendered_images

    @staticmethod
    def save_image(image: np.ndarray, path: str) -> None:
        """
        Save rendered image to disk

        Args:
            image: HxWx3 numpy array (0-1 range)
            path: output path
        """
        # Convert to 0-255 range
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        # Save
        pil_image = PILImage.fromarray(image_uint8)
        pil_image.save(path)
