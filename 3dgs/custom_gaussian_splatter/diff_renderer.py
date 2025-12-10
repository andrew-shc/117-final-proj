"""
Differentiable Gaussian Splatting Renderer in PyTorch
Enables gradient flow for training
"""

from typing import Tuple, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from gaussian_model import GSplats
    import pycolmap


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices (differentiable)

    Args:
        quaternions: Nx4 tensor [w, x, y, z] (scalar-first convention)

    Returns:
        rotation_matrices: Nx3x3 rotation matrices
    """
    # Normalize quaternions
    q = F.normalize(quaternions, p=2, dim=1)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Build rotation matrices
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    R = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)

    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)

    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y*z - w*x)

    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return R


def build_covariance_3d_torch(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """
    Build 3D covariance matrices from scales and rotations (differentiable)

    Covariance formula: Σ = R @ S @ S^T @ R^T
    where R is the rotation matrix (world space orientation)
    and S is the diagonal scale matrix

    This builds the covariance in world space. To transform to camera space,
    multiply by camera rotation: Σ_cam = R_cam @ Σ_world @ R_cam^T

    Args:
        scales: Nx3 scale parameters (world space)
        rotations: Nx4 quaternions [w, x, y, z] (world space orientation)

    Returns:
        cov: Nx3x3 covariance matrices (world space)
    """
    # Convert quaternions to rotation matrices (world space)
    R = quaternion_to_rotation_matrix(rotations)  # Nx3x3

    # Build scale matrices (diagonal)
    S = torch.diag_embed(scales)  # Nx3x3

    # Covariance in world space: Σ = R @ S @ S^T @ R^T
    RS = torch.bmm(R, S)  # Nx3x3
    cov = torch.bmm(RS, RS.transpose(1, 2))  # Nx3x3

    return cov


class DifferentiableRasterizer:
    """
    Differentiable Gaussian Splatting Rasterizer in PyTorch
    Slower but enables gradient flow for training
    """

    def __init__(self, gaussians: 'GSplats'):
        """
        Initialize differentiable rasterizer

        Args:
            gaussians: GSplats object containing Gaussian parameters
        """
        self.gaussians = gaussians
        
        # For tracking view-space positional gradients (used for densification)
        self.viewspace_point_gradients = None
        self.viewspace_point_gradient_accum = None
        self.gradient_accum_count = 0

    def rasterize(
        self,
        camera: 'pycolmap.Camera',
        image: 'pycolmap.Image',
        width: int = None,
        height: int = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Render Gaussians for a single camera pose (differentiable)

        Args:
            camera: pycolmap Camera object
            image: pycolmap Image object (contains camera pose)
            width: output width (default: camera width)
            height: output height (default: camera height)
            verbose: print rendering statistics

        Returns:
            rendered: HxWx3 RGB image tensor (0-1 range)
        """
        if width is None:
            width = camera.width
        if height is None:
            height = camera.height

        device = self.gaussians.device

        # Compute resolution scale factor
        scale_factor = width / camera.width

        # Get Gaussian parameters (already torch tensors with gradients)
        positions = self.gaussians.positions  # Nx3
        colors = self.gaussians.colors  # Nx3
        scales = self.gaussians.scales  # Nx3
        rotations = self.gaussians.rotations  # Nx4
        opacities = self.gaussians.opacities  # Nx1

        # Get camera pose
        cam_from_world = image.cam_from_world()
        R_cam = torch.tensor(cam_from_world.rotation.matrix(), dtype=torch.float32, device=device)  # 3x3
        t_cam = torch.tensor(cam_from_world.translation, dtype=torch.float32, device=device)  # 3

        # Transform to camera space
        means_cam = (R_cam @ positions.T).T + t_cam  # Nx3
        depths = means_cam[:, 2]  # N

        # Filter points behind camera
        valid_mask = depths > 0.1

        # Early exit if no valid Gaussians
        if valid_mask.sum() == 0:
            return torch.zeros((height, width, 3), dtype=torch.float32, device=device)

        # Get camera intrinsics and scale them
        fx, fy, cx, cy = camera.params[:4]
        fx_scaled = fx * scale_factor
        fy_scaled = fy * scale_factor
        cx_scaled = cx * scale_factor
        cy_scaled = cy * scale_factor

        # Project to 2D with scaled intrinsics
        # Store means_2d with gradient tracking for view-space gradients
        means_2d_x = (fx_scaled * means_cam[:, 0] / means_cam[:, 2]) + cx_scaled
        means_2d_y = (fy_scaled * means_cam[:, 1] / means_cam[:, 2]) + cy_scaled
        means_2d = torch.stack([means_2d_x, means_2d_y], dim=1)  # Nx2
        
        # Enable gradient tracking for view-space positions (for densification)
        if means_2d.requires_grad:
            means_2d.retain_grad()
            self.viewspace_point_gradients = means_2d

        # Check image bounds
        valid_mask = valid_mask & (means_2d[:, 0] >= 0) & (means_2d[:, 0] < width)
        valid_mask = valid_mask & (means_2d[:, 1] >= 0) & (means_2d[:, 1] < height)

        if valid_mask.sum() == 0:
            return torch.zeros((height, width, 3), dtype=torch.float32, device=device)

        # Filter to valid Gaussians
        means_cam_valid = means_cam[valid_mask]
        means_2d_valid = means_2d[valid_mask]
        colors_valid = colors[valid_mask]
        opacities_valid = opacities[valid_mask].squeeze(1)
        depths_valid = depths[valid_mask]
        scales_valid = scales[valid_mask]
        rotations_valid = rotations[valid_mask]

        # Build 3D covariances in world space
        # Each Gaussian: Σ_world = R_gaussian @ S @ S^T @ R_gaussian^T
        cov_3d_world = build_covariance_3d_torch(scales_valid, rotations_valid)  # Mx3x3

        # Transform covariance to camera space
        # Σ_cam = R_cam @ Σ_world @ R_cam^T
        # This correctly transforms the orientation from world to camera coordinates
        R_cam_expanded = R_cam.unsqueeze(0).expand(cov_3d_world.shape[0], -1, -1)
        cov_cam = torch.bmm(torch.bmm(R_cam_expanded, cov_3d_world), R_cam_expanded.transpose(1, 2))

        # Jacobian of perspective projection (use scaled intrinsics)
        z = means_cam_valid[:, 2:3]  # Mx1
        J = torch.zeros((means_cam_valid.shape[0], 2, 3), device=device)
        J[:, 0, 0] = fx_scaled / z[:, 0]
        J[:, 0, 2] = -fx_scaled * means_cam_valid[:, 0] / (z[:, 0] ** 2)
        J[:, 1, 1] = fy_scaled / z[:, 0]
        J[:, 1, 2] = -fy_scaled * means_cam_valid[:, 1] / (z[:, 0] ** 2)

        # Project covariance to 2D: cov_2d = J @ cov_cam @ J^T
        cov_2d = torch.bmm(torch.bmm(J, cov_cam), J.transpose(1, 2))  # Mx2x2

        # Add regularization for numerical stability
        cov_2d = cov_2d + torch.eye(2, device=device).unsqueeze(0) * 1e-4

        # Sort by depth (back to front)
        depth_order = torch.argsort(depths_valid)
        means_2d_sorted = means_2d_valid[depth_order]
        cov_2d_sorted = cov_2d[depth_order]
        colors_sorted = colors_valid[depth_order]
        opacities_sorted = opacities_valid[depth_order]

        # Render using differentiable splatting
        rendered = self._render_gaussians_to_grid(
            means_2d_sorted, cov_2d_sorted, colors_sorted, opacities_sorted,
            height, width, device
        )

        return rendered

    def _render_gaussians_to_grid(
        self,
        means_2d: torch.Tensor,
        cov_2d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        height: int,
        width: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Render Gaussians to image grid (differentiable but simplified)

        This is a simplified version that trades accuracy for differentiability
        """
        # Create pixel grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_grid, y_grid], dim=-1)  # HxWx2

        # Initialize output
        rendered = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
        alpha_accumulated = torch.zeros((height, width), dtype=torch.float32, device=device)

        # Render each Gaussian (back to front)
        for i in range(means_2d.shape[0]):
            mean = means_2d[i]  # 2
            cov = cov_2d[i]  # 2x2
            color = colors[i]  # 3
            opacity = opacities[i]  # scalar

            # Compute inverse covariance
            cov_inv = torch.linalg.inv(cov)

            # Determine bounding box (3 sigma rule)
            eigenvalues = torch.linalg.eigvalsh(cov)
            radius = 3.0 * torch.sqrt(eigenvalues.max())

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

            # Compute Mahalanobis distance
            # d^2 = delta^T @ cov_inv @ delta
            delta_flat = delta.reshape(-1, 2)  # (H*W)x2
            mahal = torch.sum((delta_flat @ cov_inv) * delta_flat, dim=1)  # H*W
            mahal = mahal.reshape(delta.shape[0], delta.shape[1])  # HxW

            # Gaussian weight
            gaussian_weight = torch.exp(-0.5 * mahal)

            # Alpha blending
            alpha = opacity * gaussian_weight
            transmittance = 1.0 - alpha_accumulated[y_min:y_max, x_min:x_max]
            alpha_blend = alpha * transmittance

            # Accumulate color
            rendered[y_min:y_max, x_min:x_max] += alpha_blend.unsqueeze(-1) * color

            # Update alpha accumulation
            alpha_accumulated[y_min:y_max, x_min:x_max] += alpha_blend

        # Clamp to valid range
        rendered = torch.clamp(rendered, 0, 1)

        return rendered

    def update_gradient_accum(self):
        """
        Accumulate view-space positional gradients for densification.
        Call this after loss.backward() during training.
        """
        if self.viewspace_point_gradients is None or self.viewspace_point_gradients.grad is None:
            return
        
        # Get gradients (Nx2 for 2D screen positions)
        grads = self.viewspace_point_gradients.grad
        
        # Compute magnitude of gradients
        grad_norms = torch.norm(grads, dim=1, keepdim=True)  # Nx1
        
        # Initialize accumulator if needed
        if self.viewspace_point_gradient_accum is None:
            self.viewspace_point_gradient_accum = torch.zeros(
                (self.gaussians.num_gaussians, 1),
                dtype=torch.float32,
                device=self.gaussians.device
            )
        
        # Accumulate gradients (add to existing Gaussians)
        self.viewspace_point_gradient_accum += grad_norms
        self.gradient_accum_count += 1
    
    def get_average_gradient_norm(self):
        """
        Get average view-space gradient magnitude per Gaussian.
        Used to identify which Gaussians need densification.
        
        Returns:
            avg_grads: Nx1 tensor of average gradient magnitudes
        """
        if self.viewspace_point_gradient_accum is None or self.gradient_accum_count == 0:
            return torch.zeros((self.gaussians.num_gaussians, 1), device=self.gaussians.device)
        
        return self.viewspace_point_gradient_accum / self.gradient_accum_count
    
    def reset_gradient_accum(self):
        """
        Reset gradient accumulation counters.
        Call this after densification to start fresh.
        """
        self.viewspace_point_gradient_accum = None
        self.gradient_accum_count = 0
        self.viewspace_point_gradients = None
