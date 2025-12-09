import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData

class GaussianSplatVAE(nn.Module):
    """
    VAE for compressing 3D Gaussian Splats to 2D grid representation.
    
    Input: 3D GS point cloud with attributes (xyz, sh_dc, opacity, scale, rotation)
    Output: 2D grid of shape (12, H, W) encoding Gaussian parameters
    
    Grid channels (12 total):
    - [0:3]: xyz position deltas (relative to grid cell center)
    - [3:6]: sh_dc color coefficients
    - [6]: opacity
    - [7:10]: scale (log space)
    - [10:14]: rotation quaternion (will be normalized)
    
    Actually 14 channels, adjusting:
    - [0:3]: xyz deltas
    - [3:6]: sh_dc
    - [6]: opacity  
    - [7:10]: scale
    - [10:14]: rotation (4D quaternion)
    """
    
    def __init__(self, grid_h=64, grid_w=64, latent_dim=512, hidden_dim=1024):  # INCREASED SIZE
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.latent_dim = latent_dim
        self.grid_channels = 14  # xyz(3) + sh_dc(3) + opacity(1) + scale(3) + rot(4)
        
        # Encoder: 3D point cloud → latent code (DEEPER & WIDER)
        # Uses PointNet-style architecture for permutation invariance
        self.point_encoder = nn.Sequential(
            nn.Linear(13, hidden_dim),  # input: xyz(3) + sh_dc(3) + opacity(1) + scale(3) + rot(3, ignore w)
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Global pooling encoder (DEEPER)
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2),  # mean and logvar
        )
        
        # Decoder: latent → 2D grid (DEEPER & WIDER)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
        )
        
        # Upsample to grid resolution (MORE CHANNELS)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 512, 4, 2, 1),  # 2x2 -> 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.grid_channels, 4, 2, 1),  # 32x32 -> 64x64
        )
        
    def encode(self, xyz, sh_dc, opacity, scale, rotation):
        """
        Encode 3D GS point cloud to latent distribution.
        
        Args:
            xyz: (B, N, 3) positions
            sh_dc: (B, N, 3) SH DC coefficients
            opacity: (B, N, 1) opacity values
            scale: (B, N, 3) scale parameters
            rotation: (B, N, 4) quaternion rotations
        
        Returns:
            mu: (B, latent_dim) mean
            logvar: (B, latent_dim) log variance
        """
        # Normalize inputs
        xyz_norm = (xyz - xyz.mean(dim=1, keepdim=True)) / (xyz.std(dim=1, keepdim=True) + 1e-8)
        
        # Concatenate features (ignore rotation w component for now, use xyz)
        features = torch.cat([
            xyz_norm, 
            sh_dc, 
            opacity, 
            scale,
            rotation[:, :, :3]  # Use first 3 components of quaternion
        ], dim=-1)  # (B, N, 13)
        
        # Point-wise encoding
        point_features = self.point_encoder(features)  # (B, N, hidden_dim)
        
        # Max pooling for permutation invariance
        global_features = point_features.max(dim=1)[0]  # (B, hidden_dim)
        
        # Encode to latent distribution
        latent_params = self.global_encoder(global_features)  # (B, latent_dim * 2)
        mu, logvar = latent_params.chunk(2, dim=-1)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, original_xyz_stats=None):
        """
        Decode latent code to 2D grid.
        
        Args:
            z: (B, latent_dim) latent code
            original_xyz_stats: Optional tuple (mean, std) for denormalizing xyz deltas
        
        Returns:
            grid: (B, 14, H, W) parameter grid
        """
        B = z.shape[0]
        
        # Decode to feature map
        features = self.decoder_fc(z)  # (B, hidden_dim * 4)
        features = features.view(B, -1, 2, 2)  # (B, hidden_dim, 2, 2)
        
        # Upsample to grid
        grid = self.decoder_conv(features)  # (B, 14, H, W)
        
        # Apply appropriate activations/constraints per channel type
        xyz_delta = torch.tanh(grid[:, 0:3]) * 2.0  # Scale to [-2, 2] for better range
        sh_dc = torch.tanh(grid[:, 3:6])  # [-1, 1] range
        opacity = torch.sigmoid(grid[:, 6:7])  # [0, 1]
        scale = torch.clamp(grid[:, 7:10], min=-10, max=10)  # Clamp log scale to prevent overflow
        rotation = F.normalize(grid[:, 10:14], dim=1, eps=1e-6)  # Normalize quaternion with epsilon
        
        grid = torch.cat([xyz_delta, sh_dc, opacity, scale, rotation], dim=1)
        
        return grid
    
    def forward(self, xyz, sh_dc, opacity, scale, rotation):
        """
        Full forward pass through VAE.
        
        Returns:
            grid: (B, 14, H, W) reconstructed grid
            mu: (B, latent_dim) latent mean
            logvar: (B, latent_dim) latent log variance
        """
        # Encode
        mu, logvar = self.encode(xyz, sh_dc, opacity, scale, rotation)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        xyz_mean = xyz.mean(dim=1)
        xyz_std = xyz.std(dim=1)
        grid = self.decode(z, (xyz_mean, xyz_std))
        
        return grid, mu, logvar
    
    def grid_to_gaussians(self, grid, num_samples_per_cell=4):
        """
        Convert 2D grid back to 3D Gaussian Splats.
        
        Args:
            grid: (B, 14, H, W) parameter grid
            num_samples_per_cell: Number of Gaussians to sample per grid cell
        
        Returns:
            xyz, sh_dc, opacity, scale, rotation tensors
        """
        B, C, H, W = grid.shape
        device = grid.device
        
        # Create grid cell centers in normalized space [-1, 1]
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # Sample multiple points per cell with small jitter
        all_xyz = []
        all_sh_dc = []
        all_opacity = []
        all_scale = []
        all_rotation = []
        
        for i in range(num_samples_per_cell):
            # Add jitter to grid positions
            jitter = torch.randn(2, H, W, device=device) * 0.05  # Smaller jitter
            x_pos = x_grid + jitter[0] / W
            y_pos = y_grid + jitter[1] / H
            
            # Extract parameters from grid
            xyz_delta = grid[:, 0:3].permute(0, 2, 3, 1)  # (B, H, W, 3)
            sh_dc = grid[:, 3:6].permute(0, 2, 3, 1)
            opacity = grid[:, 6:7].permute(0, 2, 3, 1)
            scale_log = grid[:, 7:10].permute(0, 2, 3, 1)
            scale = torch.exp(torch.clamp(scale_log, min=-10, max=5))  # Clamp before exp to prevent overflow
            rotation = grid[:, 10:14].permute(0, 2, 3, 1)
            
            # Compute absolute xyz positions
            z_pos = torch.zeros_like(x_pos)
            grid_centers = torch.stack([x_pos, y_pos, z_pos], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            xyz = grid_centers + xyz_delta
            
            all_xyz.append(xyz.reshape(B, -1, 3))
            all_sh_dc.append(sh_dc.reshape(B, -1, 3))
            all_opacity.append(opacity.reshape(B, -1, 1))
            all_scale.append(scale.reshape(B, -1, 3))
            all_rotation.append(rotation.reshape(B, -1, 4))
        
        # Concatenate all samples
        xyz = torch.cat(all_xyz, dim=1)
        sh_dc = torch.cat(all_sh_dc, dim=1)
        opacity = torch.cat(all_opacity, dim=1)
        scale = torch.cat(all_scale, dim=1)
        rotation = torch.cat(all_rotation, dim=1)
        
        return xyz, sh_dc, opacity, scale, rotation


def chamfer_distance_manual(x, y):
    """
    Compute Chamfer Distance without PyTorch3D.
    
    Args:
        x: (B, N, 3) tensor
        y: (B, M, 3) tensor
    
    Returns:
        chamfer_dist: scalar tensor
    """
    # x: (B, N, 3), y: (B, M, 3)
    # Compute pairwise distances
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T y
    
    xx = (x ** 2).sum(dim=2)  # (B, N)
    yy = (y ** 2).sum(dim=2)  # (B, M)
    xy = torch.bmm(x, y.transpose(1, 2))  # (B, N, M)
    
    # Pairwise squared distances (clamp to avoid negative values from numerical errors)
    dist = xx.unsqueeze(2) + yy.unsqueeze(1) - 2 * xy  # (B, N, M)
    dist = torch.clamp(dist, min=0.0)  # Ensure non-negative
    
    # Find nearest neighbors
    min_dist_xy = dist.min(dim=2)[0]  # (B, N) - for each point in x, distance to nearest in y
    min_dist_yx = dist.min(dim=1)[0]  # (B, M) - for each point in y, distance to nearest in x
    
    # Chamfer distance is the sum of both directions
    chamfer = min_dist_xy.mean() + min_dist_yx.mean()
    
    return chamfer


def vae_loss(recon_xyz, recon_sh_dc, recon_opacity, recon_scale, recon_rotation,
             target_xyz, target_sh_dc, target_opacity, target_scale, target_rotation,
             mu, logvar, kl_weight=0.001):
    """
    Compute VAE loss with reconstruction and KL divergence terms.
    
    Uses manual Chamfer Distance for geometry and L2 for attributes.
    """
    # Check for NaNs in inputs
    if torch.isnan(recon_xyz).any() or torch.isnan(target_xyz).any():
        print("WARNING: NaN detected in xyz inputs!")
        return torch.tensor(float('nan')), {}
    
    # Chamfer distance for positions (no PyTorch3D needed)
    chamfer_loss = chamfer_distance_manual(recon_xyz, target_xyz)
    
    if torch.isnan(chamfer_loss):
        print("WARNING: NaN in chamfer loss!")
        chamfer_loss = torch.tensor(0.0, device=recon_xyz.device)
    
    # For attributes, match to nearest neighbors first
    # Compute nearest neighbor mapping
    with torch.no_grad():
        xx = (recon_xyz ** 2).sum(dim=2)
        yy = (target_xyz ** 2).sum(dim=2)
        xy = torch.bmm(recon_xyz, target_xyz.transpose(1, 2))
        dist = torch.clamp(xx.unsqueeze(2) + yy.unsqueeze(1) - 2 * xy, min=0.0)
        nearest_idx = dist.argmin(dim=2)  # (B, N_recon)
    
    # Gather nearest target attributes
    B, N_recon = nearest_idx.shape
    N_target = target_xyz.shape[1]
    
    # Expand indices for gathering
    batch_idx = torch.arange(B, device=nearest_idx.device).unsqueeze(1).expand(-1, N_recon)
    
    target_sh_dc_matched = target_sh_dc[batch_idx, nearest_idx]
    target_opacity_matched = target_opacity[batch_idx, nearest_idx]
    target_scale_matched = target_scale[batch_idx, nearest_idx]
    target_rotation_matched = target_rotation[batch_idx, nearest_idx]
    
    # L2 loss for attributes with clamping
    sh_dc_loss = F.mse_loss(recon_sh_dc, target_sh_dc_matched)
    opacity_loss = F.mse_loss(recon_opacity, target_opacity_matched)
    
    # Scale loss with better stability
    recon_scale_clamped = torch.clamp(recon_scale, min=1e-6, max=100)
    target_scale_clamped = torch.clamp(target_scale_matched, min=1e-6, max=100)
    scale_loss = F.mse_loss(torch.log(recon_scale_clamped), torch.log(target_scale_clamped))
    
    # Rotation loss with epsilon
    rotation_sim = F.cosine_similarity(recon_rotation, target_rotation_matched, dim=-1, eps=1e-6)
    rotation_loss = 1 - rotation_sim.mean()
    
    # Check for NaNs in individual losses
    if torch.isnan(sh_dc_loss):
        print("WARNING: NaN in sh_dc_loss")
        sh_dc_loss = torch.tensor(0.0, device=recon_xyz.device)
    if torch.isnan(opacity_loss):
        print("WARNING: NaN in opacity_loss")
        opacity_loss = torch.tensor(0.0, device=recon_xyz.device)
    if torch.isnan(scale_loss):
        print("WARNING: NaN in scale_loss")
        scale_loss = torch.tensor(0.0, device=recon_xyz.device)
    if torch.isnan(rotation_loss):
        print("WARNING: NaN in rotation_loss")
        rotation_loss = torch.tensor(0.0, device=recon_xyz.device)
    
    # Total reconstruction loss
    recon_loss = (
        chamfer_loss + 
        sh_dc_loss + 
        opacity_loss + 
        scale_loss + 
        rotation_loss
    )
    
    # KL divergence with clamping
    kl_loss = -0.5 * torch.sum(1 + torch.clamp(logvar, min=-10, max=10) - mu.pow(2) - torch.clamp(logvar, min=-10, max=10).exp()) / mu.shape[0]
    kl_loss = torch.clamp(kl_loss, min=0.0, max=1000.0)  # Prevent extreme values
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'chamfer': chamfer_loss.item(),
        'sh_dc': sh_dc_loss.item(),
        'opacity': opacity_loss.item(),
        'scale': scale_loss.item(),
        'rotation': rotation_loss.item(),
        'kl': kl_loss.item(),
    }


# Example usage
def load_3dgs_ply(path):
    """Load standard 3D Gaussian Splatting PLY format"""
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    N = len(vertex)
    
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1).astype(np.float32)
    opacity = vertex['opacity'].astype(np.float32)
    scale = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1).astype(np.float32)
    rotation = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1).astype(np.float32)
    
    return xyz, sh_dc, opacity, scale, rotation


class GaussianSplatDataset(torch.utils.data.Dataset):
    """Dataset for loading multiple 3DGS PLY files."""
    
    def __init__(self, ply_paths, max_points=8192):
        self.ply_paths = ply_paths
        self.max_points = max_points
        
    def __len__(self):
        return len(self.ply_paths)
    
    def __getitem__(self, idx):
        xyz, sh_dc, opacity, scale, rotation = load_3dgs_ply(self.ply_paths[idx])
        
        # Subsample or pad to fixed size
        N = len(xyz)
        if N > self.max_points:
            indices = np.random.choice(N, self.max_points, replace=False)
            xyz = xyz[indices]
            sh_dc = sh_dc[indices]
            opacity = opacity[indices]
            scale = scale[indices]
            rotation = rotation[indices]
        elif N < self.max_points:
            # Pad with duplicate points
            pad_size = self.max_points - N
            pad_indices = np.random.choice(N, pad_size, replace=True)
            xyz = np.concatenate([xyz, xyz[pad_indices]], axis=0)
            sh_dc = np.concatenate([sh_dc, sh_dc[pad_indices]], axis=0)
            opacity = np.concatenate([opacity, opacity[pad_indices]], axis=0)
            scale = np.concatenate([scale, scale[pad_indices]], axis=0)
            rotation = np.concatenate([rotation, rotation[pad_indices]], axis=0)
        
        return {
            'xyz': torch.from_numpy(xyz),
            'sh_dc': torch.from_numpy(sh_dc),
            'opacity': torch.from_numpy(opacity).unsqueeze(-1),
            'scale': torch.from_numpy(scale),
            'rotation': torch.from_numpy(rotation),
        }


def train_vae(model, train_loader, num_epochs=100, lr=1e-4, device='cuda', 
              kl_weight_schedule='warmup', save_path='vae_checkpoint.pt'):
    """
    Train the VAE with KL annealing and save checkpoints.
    
    Args:
        kl_weight_schedule: 'warmup' (gradual increase) or 'constant'
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {
            'total': 0, 'recon': 0, 'chamfer': 0, 
            'sh_dc': 0, 'opacity': 0, 'scale': 0, 'rotation': 0, 'kl': 0
        }
        
        # KL weight annealing
        if kl_weight_schedule == 'warmup':
            kl_weight = min(0.01, 0.0001 * (epoch + 1))  # Gradually increase
        else:
            kl_weight = 0.001
        
        for batch_idx, batch in enumerate(train_loader):
            xyz = batch['xyz'].to(device)
            sh_dc = batch['sh_dc'].to(device)
            opacity = batch['opacity'].to(device)
            scale = batch['scale'].to(device)
            rotation = batch['rotation'].to(device)
            
            # Debug: Check input data
            if batch_idx == 0 and epoch == 0:
                print("\n=== Input Data Check ===")
                print(f"xyz range: [{xyz.min().item():.3f}, {xyz.max().item():.3f}]")
                print(f"sh_dc range: [{sh_dc.min().item():.3f}, {sh_dc.max().item():.3f}]")
                print(f"opacity range: [{opacity.min().item():.3f}, {opacity.max().item():.3f}]")
                print(f"scale range: [{scale.min().item():.3f}, {scale.max().item():.3f}]")
                print(f"rotation range: [{rotation.min().item():.3f}, {rotation.max().item():.3f}]")
                print(f"Any NaN in inputs: {torch.isnan(xyz).any() or torch.isnan(sh_dc).any()}")
            
            # Forward pass
            grid, mu, logvar = model(xyz, sh_dc, opacity, scale, rotation)
            

            
            # Decode back to 3D
            recon_xyz, recon_sh_dc, recon_opacity, recon_scale, recon_rotation = \
                model.grid_to_gaussians(grid, num_samples_per_cell=1)
            

            
            # Compute loss
            loss, loss_dict = vae_loss(
                recon_xyz, recon_sh_dc, recon_opacity, recon_scale, recon_rotation,
                xyz, sh_dc, opacity, scale, rotation,
                mu, logvar, kl_weight=kl_weight
            )
            
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN in gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
                    has_nan_grad = True
            
            if has_nan_grad:
                print("Skipping update due to NaN gradients")
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Accumulate losses
            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | KL_weight: {kl_weight:.6f}")
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
        
        scheduler.step()
        if epoch%300==0:
            print(f"\n=== Epoch {epoch+1} Summary ===")
            print(f"Total: {epoch_losses['total']:.4f} | Recon: {epoch_losses['recon']:.4f}")
            print(f"Chamfer: {epoch_losses['chamfer']:.4f} | KL: {epoch_losses['kl']:.4f}")
            print(f"SH_DC: {epoch_losses['sh_dc']:.4f} | Opacity: {epoch_losses['opacity']:.4f}")
            print(f"Scale: {epoch_losses['scale']:.4f} | Rotation: {epoch_losses['rotation']:.4f}\n")
        
        # Save best model
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"✓ Saved checkpoint to {save_path}\n")
    
    return model


def save_ply(path, xyz, sh_dc, opacity, scale, rotation):
    """Save reconstructed Gaussians back to PLY format."""
    from plyfile import PlyElement, PlyData
    
    N = len(xyz)
    
    # Prepare vertex data
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]
    
    # Convert to numpy arrays
    if torch.is_tensor(xyz):
        xyz = xyz.cpu().numpy()
        sh_dc = sh_dc.cpu().numpy()
        opacity = opacity.cpu().numpy().squeeze()
        scale = scale.cpu().numpy()
        rotation = rotation.cpu().numpy()
    
    # Create structured array
    elements = np.empty(N, dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['f_dc_0'] = sh_dc[:, 0]
    elements['f_dc_1'] = sh_dc[:, 1]
    elements['f_dc_2'] = sh_dc[:, 2]
    elements['opacity'] = opacity
    elements['scale_0'] = scale[:, 0]
    elements['scale_1'] = scale[:, 1]
    elements['scale_2'] = scale[:, 2]
    elements['rot_0'] = rotation[:, 0]
    elements['rot_1'] = rotation[:, 1]
    elements['rot_2'] = rotation[:, 2]
    elements['rot_3'] = rotation[:, 3]
    
    # Create and save PLY
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    print(f"✓ Saved {N:,} splats to {path}")


if __name__ == "__main__":
    # Configuration for LARGER model with better capacity
    GRID_SIZE = 128  # Larger grid: 64x64 for more detail
    LATENT_DIM = 512  # Larger latent: 512D
    HIDDEN_DIM = 1024  # Larger network: 1024D
    MAX_POINTS = 30000  # More points for better coverage
    
    # Single file training
    ply_path = "/home/lu/Documents/117-final-proj/3DGS_PLY_sample_data/PLY(postshot)/cactus_splat3_30kSteps_142k_splats.ply"
    
    print("=== Loading Single PLY File ===")
    xyz_orig, sh_dc_orig, opacity_orig, scale_orig, rotation_orig = load_3dgs_ply(ply_path)
    
    # Create dataset with single file repeated (for DataLoader compatibility)
    dataset = GaussianSplatDataset([ply_path], max_points=MAX_POINTS)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # Create SMALL model for diffusion
    model = GaussianSplatVAE(
        grid_h=GRID_SIZE, 
        grid_w=GRID_SIZE, 
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM
    )
    
    print(f"\n=== Model Configuration ===")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Latent dim: {LATENT_DIM}")
    print(f"Grid shape for diffusion: (14, {GRID_SIZE}, {GRID_SIZE})")
    print(f"Total grid elements: {14 * GRID_SIZE * GRID_SIZE}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Train on single file (overfitting is OK - we're learning the compression)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}\n")
    
    trained_model = train_vae(
        model, 
        train_loader, 
        num_epochs=2000,  # More epochs since single file
        lr=1e-4,
        device=device,
        kl_weight_schedule='warmup',
        save_path='gs_vae_cactus.pt'
    )
    
    # === RECONSTRUCTION ===
    print("\n" + "="*60)
    print("=== RECONSTRUCTION TEST ===")
    print("="*60 + "\n")
    
    model.eval()
    with torch.no_grad():
        # Load the full original data
        sample = dataset[0]
        xyz = sample['xyz'].unsqueeze(0).to(device)
        sh_dc = sample['sh_dc'].unsqueeze(0).to(device)
        opacity = sample['opacity'].unsqueeze(0).to(device)
        scale = sample['scale'].unsqueeze(0).to(device)
        rotation = sample['rotation'].unsqueeze(0).to(device)
        
        # Encode to grid
        grid, mu, logvar = model(xyz, sh_dc, opacity, scale, rotation)

        
        # Decode back to 3D Gaussians
        recon_xyz, recon_sh_dc, recon_opacity, recon_scale, recon_rotation = \
            model.grid_to_gaussians(grid, num_samples_per_cell=2)
        

        
        # Compute reconstruction metrics
        chamfer = chamfer_distance_manual(recon_xyz, xyz)
        
        sh_dc_error = F.mse_loss(recon_sh_dc[:, :xyz.shape[1]], sh_dc).item()
        opacity_error = F.mse_loss(recon_opacity[:, :xyz.shape[1]], opacity).item()

        # Save reconstructed PLY
        output_path = "cactus_reconstructed.ply"
        save_ply(
            output_path,
            recon_xyz[0],
            recon_sh_dc[0],
            recon_opacity[0],
            recon_scale[0],
            recon_rotation[0]
        )
        
        print(f"\n✓ Reconstruction complete!")
        print(f"✓ Original: {ply_path}")
        print(f"✓ Reconstructed: {output_path}")
        print(f"\n✓ Grid representation (14, {GRID_SIZE}, {GRID_SIZE}) ready for diffusion!")