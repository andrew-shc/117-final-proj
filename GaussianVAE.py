import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData, PlyElement

class GaussianSplatVAE(nn.Module):
    def __init__(self, grid_h=64, grid_w=64, latent_dim=256, hidden_dim=512):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.latent_dim = latent_dim
        self.grid_channels = 14
        self.point_encoder = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
        )
        if grid_h == 48:
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, 3, 0),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, self.grid_channels, 3, 1, 1),
            )
        else:
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, self.grid_channels, 4, 2, 1),
            )
    def encode(self, xyz, sh_dc, opacity, scale, rotation):
        xyz_norm = (xyz - xyz.mean(dim=1, keepdim=True)) / (xyz.std(dim=1, keepdim=True) + 1e-8)
        features = torch.cat([xyz_norm, sh_dc, opacity, scale, rotation[:, :, :3]], dim=-1)
        point_features = self.point_encoder(features)
        global_features = point_features.max(dim=1)[0]
        latent_params = self.global_encoder(global_features)
        mu, logvar = latent_params.chunk(2, dim=-1)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z, original_xyz_stats=None):
        B = z.shape[0]
        features = self.decoder_fc(z)
        features = features.view(B, -1, 2, 2)
        grid = self.decoder_conv(features)
        xyz_delta = torch.tanh(grid[:, 0:3]) * 2.0
        sh_dc = torch.tanh(grid[:, 3:6])
        opacity = torch.sigmoid(grid[:, 6:7])
        scale = torch.clamp(grid[:, 7:10], min=-10, max=10)
        rotation = F.normalize(grid[:, 10:14], dim=1, eps=1e-6)
        grid = torch.cat([xyz_delta, sh_dc, opacity, scale, rotation], dim=1)
        return grid
    def forward(self, xyz, sh_dc, opacity, scale, rotation):
        mu, logvar = self.encode(xyz, sh_dc, opacity, scale, rotation)
        z = self.reparameterize(mu, logvar)
        xyz_mean = xyz.mean(dim=1)
        xyz_std = xyz.std(dim=1)
        grid = self.decode(z, (xyz_mean, xyz_std))
        return grid, mu, logvar
    def grid_to_gaussians(self, grid, num_samples_per_cell=4):
        B, C, H, W = grid.shape
        device = grid.device
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        all_xyz = []
        all_sh_dc = []
        all_opacity = []
        all_scale = []
        all_rotation = []
        for i in range(num_samples_per_cell):
            jitter = torch.randn(2, H, W, device=device) * 0.05
            x_pos = x_grid + jitter[0] / W
            y_pos = y_grid + jitter[1] / H
            xyz_delta = grid[:, 0:3].permute(0, 2, 3, 1)
            sh_dc = grid[:, 3:6].permute(0, 2, 3, 1)
            opacity = grid[:, 6:7].permute(0, 2, 3, 1)
            scale_log = grid[:, 7:10].permute(0, 2, 3, 1)
            scale = torch.exp(torch.clamp(scale_log, min=-10, max=5))
            rotation = grid[:, 10:14].permute(0, 2, 3, 1)
            z_pos = torch.zeros_like(x_pos)
            grid_centers = torch.stack([x_pos, y_pos, z_pos], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            xyz = grid_centers + xyz_delta
            all_xyz.append(xyz.reshape(B, -1, 3))
            all_sh_dc.append(sh_dc.reshape(B, -1, 3))
            all_opacity.append(opacity.reshape(B, -1, 1))
            all_scale.append(scale.reshape(B, -1, 3))
            all_rotation.append(rotation.reshape(B, -1, 4))
        xyz = torch.cat(all_xyz, dim=1)
        sh_dc = torch.cat(all_sh_dc, dim=1)
        opacity = torch.cat(all_opacity, dim=1)
        scale = torch.cat(all_scale, dim=1)
        rotation = torch.cat(all_rotation, dim=1)
        return xyz, sh_dc, opacity, scale, rotation

def chamfer_distance_manual(x, y):
    xx = (x**2).sum(dim=2)
    yy = (y**2).sum(dim=2)
    xy = torch.bmm(x,y.transpose(1, 2))
    dist = xx.unsqueeze(2) + yy.unsqueeze(1)-2 * xy
    dist = torch.clamp(dist, min=0.0)
    min_dist_xy = dist.min(dim=2)[0]
    min_dist_yx = dist.min(dim=1)[0]
    chamfer = min_dist_xy.mean()+min_dist_yx.mean()
    return chamfer

def vae_loss(recon_xyz, recon_sh_dc, recon_opacity, recon_scale, recon_rotation,
             target_xyz, target_sh_dc, target_opacity, target_scale, target_rotation,
             mu, logvar, kl_weight=0.001):
    if torch.isnan(recon_xyz).any() or torch.isnan(target_xyz).any():
        return torch.tensor(float('nan')), {}
    chamfer_loss = chamfer_distance_manual(recon_xyz, target_xyz)
    if torch.isnan(chamfer_loss):
        chamfer_loss = torch.tensor(0.0, device=recon_xyz.device)
    with torch.no_grad():
        xx = (recon_xyz ** 2).sum(dim=2)
        yy = (target_xyz ** 2).sum(dim=2)
        xy = torch.bmm(recon_xyz, target_xyz.transpose(1, 2))
        dist = torch.clamp(xx.unsqueeze(2) + yy.unsqueeze(1) - 2 * xy, min=0.0)
        nearest_idx = dist.argmin(dim=2)
    B, N_recon = nearest_idx.shape
    N_target = target_xyz.shape[1]
    batch_idx = torch.arange(B, device=nearest_idx.device).unsqueeze(1).expand(-1, N_recon)
    target_sh_dc_matched = target_sh_dc[batch_idx, nearest_idx]
    target_opacity_matched = target_opacity[batch_idx, nearest_idx]
    target_scale_matched = target_scale[batch_idx, nearest_idx]
    target_rotation_matched = target_rotation[batch_idx, nearest_idx]
    sh_dc_loss = F.mse_loss(recon_sh_dc, target_sh_dc_matched)
    opacity_loss = F.mse_loss(recon_opacity, target_opacity_matched)
    recon_scale_clamped = torch.clamp(recon_scale, min=1e-6, max=100)
    target_scale_clamped = torch.clamp(target_scale_matched, min=1e-6, max=100)
    scale_loss = F.mse_loss(torch.log(recon_scale_clamped), torch.log(target_scale_clamped))
    rotation_sim = F.cosine_similarity(recon_rotation, target_rotation_matched, dim=-1, eps=1e-6)
    rotation_loss = 1 - rotation_sim.mean()
    if torch.isnan(sh_dc_loss):
        sh_dc_loss = torch.tensor(0.0, device=recon_xyz.device)
    if torch.isnan(opacity_loss):
        opacity_loss = torch.tensor(0.0, device=recon_xyz.device)
    if torch.isnan(scale_loss):
        scale_loss = torch.tensor(0.0, device=recon_xyz.device)
    if torch.isnan(rotation_loss):
        rotation_loss = torch.tensor(0.0, device=recon_xyz.device)
    recon_loss = chamfer_loss + sh_dc_loss + opacity_loss + scale_loss + rotation_loss
    kl_loss = -0.5 * torch.sum(1 + torch.clamp(logvar, min=-10, max=10) - mu.pow(2) - torch.clamp(logvar, min=-10, max=10).exp()) / mu.shape[0]
    kl_loss = torch.clamp(kl_loss, min=0.0, max=1000.0)
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

def load_3dgs_ply(path):
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    N = len(vertex)
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1).astype(np.float32)
    opacity = vertex['opacity'].astype(np.float32)
    scale = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1).astype(np.float32)
    rotation = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1).astype(np.float32)
    print(xyz.shape, sh_dc.shape, opacity.shape, scale.shape, rotation.shape)
    return xyz, sh_dc, opacity, scale, rotation

class GaussianSplatDataset(torch.utils.data.Dataset):
    def __init__(self, ply_paths, max_points=None, subsample_for_training=True):
        self.ply_paths = ply_paths
        self.max_points = max_points
        self.subsample_for_training = subsample_for_training
    def __len__(self):
        return len(self.ply_paths)
    def __getitem__(self, idx):
        xyz, sh_dc, opacity, scale, rotation = load_3dgs_ply(self.ply_paths[idx])
        N = len(xyz)
        if self.subsample_for_training and self.max_points is not None:
            if N > self.max_points:
                indices = np.random.choice(N, self.max_points, replace=False)
                xyz = xyz[indices]
                sh_dc = sh_dc[indices]
                opacity = opacity[indices]
                scale = scale[indices]
                rotation = rotation[indices]
            elif N < self.max_points:
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
        if kl_weight_schedule == 'warmup':
            kl_weight = min(0.01, 0.0001 * (epoch + 1))
        else:
            kl_weight = 0.001
        for batch_idx, batch in enumerate(train_loader):
            xyz = batch['xyz'].to(device)
            sh_dc = batch['sh_dc'].to(device)
            opacity = batch['opacity'].to(device)
            scale = batch['scale'].to(device)
            rotation = batch['rotation'].to(device)
            grid, mu, logvar = model(xyz, sh_dc, opacity, scale, rotation)
            recon_xyz, recon_sh_dc, recon_opacity, recon_scale, recon_rotation = \
                model.grid_to_gaussians(grid, num_samples_per_cell=SAMPLES_PER_CELL)
            loss, loss_dict = vae_loss(
                recon_xyz, recon_sh_dc, recon_opacity, recon_scale, recon_rotation,
                xyz, sh_dc, opacity, scale, rotation,
                mu, logvar, kl_weight=kl_weight
            )
            if torch.isnan(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
            if has_nan_grad:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}|Batch {batch_idx}/{len(train_loader)}|Loss:{loss.item():.4f}|KL:{kl_weight:.6f}")
        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
        scheduler.step()
        print(f"E{epoch+1}|Tot:{epoch_losses['total']:.4f}|Rec:{epoch_losses['recon']:.4f}|Ch:{epoch_losses['chamfer']:.4f}|KL:{epoch_losses['kl']:.4f}")
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"Saved:{save_path}")
    return model
def encode_full_gs(model, ply_path, device='cuda', chunk_size=8192):
    model.eval()
    xyz_full, sh_dc_full, opacity_full, scale_full, rotation_full = load_3dgs_ply(ply_path)
    N_full = len(xyz_full)
    all_mus = []
    all_logvars = []
    with torch.no_grad():
        for i in range(0, N_full, chunk_size):
            chunk_end = min(i + chunk_size, N_full)
            xyz = torch.from_numpy(xyz_full[i:chunk_end]).unsqueeze(0).to(device)
            sh_dc = torch.from_numpy(sh_dc_full[i:chunk_end]).unsqueeze(0).to(device)
            opacity = torch.from_numpy(opacity_full[i:chunk_end]).unsqueeze(0).unsqueeze(-1).to(device)
            scale = torch.from_numpy(scale_full[i:chunk_end]).unsqueeze(0).to(device)
            rotation = torch.from_numpy(rotation_full[i:chunk_end]).unsqueeze(0).to(device)
            mu, logvar = model.encode(xyz, sh_dc, opacity, scale, rotation)
            all_mus.append(mu.squeeze(0))
            all_logvars.append(logvar.squeeze(0))
        mu = torch.stack(all_mus, dim=0)
        logvar = torch.stack(all_logvars, dim=0)
        print(mu.shape)
    return mu, logvar

def decode_to_gs(model, mu, logvar, device='cuda', num_samples_per_cell=32):
    model.eval()
    with torch.no_grad():
        num_chunks = mu.shape[0]
        all_xyz = []
        all_sh_dc = []
        all_opacity = []
        all_scale = []
        all_rotation = []
        for i in range(num_chunks):
            mu_chunk = mu[i:i+1]
            logvar_chunk = logvar[i:i+1]
            z = model.reparameterize(mu_chunk, logvar_chunk)
            grid = model.decode(z)
            recon_xyz, recon_sh_dc, recon_opacity, recon_scale, recon_rotation = \
                model.grid_to_gaussians(grid, num_samples_per_cell=num_samples_per_cell)
            all_xyz.append(recon_xyz)
            all_sh_dc.append(recon_sh_dc)
            all_opacity.append(recon_opacity)
            all_scale.append(recon_scale)
            all_rotation.append(recon_rotation)
        recon_xyz = torch.cat(all_xyz, dim=1)
        recon_sh_dc = torch.cat(all_sh_dc, dim=1)
        recon_opacity = torch.cat(all_opacity, dim=1)
        recon_scale = torch.cat(all_scale, dim=1)
        recon_rotation = torch.cat(all_rotation, dim=1)
    return recon_xyz, recon_sh_dc, recon_opacity, recon_scale, recon_rotation
def save_ply(path, xyz, sh_dc, opacity, scale, rotation):
    N = len(xyz)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]
    if torch.is_tensor(xyz):
        xyz = xyz.cpu().numpy()
        sh_dc = sh_dc.cpu().numpy()
        opacity = opacity.cpu().numpy().squeeze()
        scale = scale.cpu().numpy()
        rotation = rotation.cpu().numpy()
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
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    print(f"Saved {N:,} splats to {path}")

if __name__ == "__main__":
    GRID_SIZE = 64
    LATENT_DIM = 768
    HIDDEN_DIM = 768
    MAX_POINTS = 4096
    SAMPLES_PER_CELL = 2
    ply_path = "/home/lu/Documents/117-final-proj/3DGS_PLY_sample_data/PLY(postshot)/cactus_splat3_30kSteps_142k_splats.ply"
    xyz_orig, sh_dc_orig, opacity_orig, scale_orig, rotation_orig = load_3dgs_ply(ply_path)
    print(xyz_orig.shape, sh_dc_orig.shape, opacity_orig.shape, scale_orig.shape, rotation_orig.shape)
    dataset = GaussianSplatDataset([ply_path], max_points=MAX_POINTS, subsample_for_training=True)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = GaussianSplatVAE(grid_h=GRID_SIZE, grid_w=GRID_SIZE, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # state_dict = torch.load('gs_vae_cactus.pt', map_location='cpu')
    # model.load_state_dict(state_dict['model_state_dict'])
    # model.to(device)
    model = train_vae(
        model, 
        train_loader, 
        num_epochs=2000, 
        lr=2e-4, 
        device=device,
        kl_weight_schedule='warmup',
        save_path='gs_vae_cactus.pt'
    )
    mu, logvar = encode_full_gs(model, ply_path, device=device, chunk_size=8192)
    print(f"Latent shape: mu={mu.shape}, logvar={logvar.shape}")
    print(f"Latent mean: {mu.mean().item():.4f} ± {mu.std().item():.4f}")
    print("Decoding latent to GS...")
    recon_xyz, recon_sh_dc, recon_opacity, recon_scale, recon_rotation = \
        decode_to_gs(model, mu, logvar, device=device, num_samples_per_cell=SAMPLES_PER_CELL)
    xyz_full, sh_dc_full, opacity_full, scale_full, rotation_full = load_3dgs_ply(ply_path)
    N_full = len(xyz_full)
    print(f"Original:{N_full:,} splats | Reconstructed:{recon_xyz.shape[1]:,} splats")
    with torch.no_grad():
        sample_size = min(8192, N_full)
        sample_indices = np.random.choice(N_full, sample_size, replace=False)
        xyz_sample = torch.from_numpy(xyz_full[sample_indices]).unsqueeze(0).to(device)
        chamfer = chamfer_distance_manual(recon_xyz[:, :sample_size], xyz_sample)
        print(f"Chamfer:{chamfer.item():.6f}")
        sh_dc_error = F.mse_loss(recon_sh_dc[:, :sample_size], torch.from_numpy(sh_dc_full[sample_indices]).unsqueeze(0).to(device)).item()
        opacity_error = F.mse_loss(recon_opacity[:, :sample_size], torch.from_numpy(opacity_full[sample_indices]).unsqueeze(0).unsqueeze(-1).to(device)).item()
        print(f"SH_DC:{sh_dc_error:.6f}|Opacity:{opacity_error:.6f}")
    output_path = "cactus_reconstructed.ply"
    print(recon_xyz.shape, recon_sh_dc.shape, recon_opacity.shape, recon_scale.shape, recon_rotation.shape)
    save_ply(output_path, recon_xyz[0], recon_sh_dc[0], recon_opacity[0], recon_scale[0], recon_rotation[0])
    print(f"Original:{ply_path}")
    print(f"Reconstructed:{output_path}")