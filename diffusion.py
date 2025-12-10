import torch
import torch.nn as nn
import math
from tqdm import tqdm
from GaussianVAE import GaussianSplatVAE, save_ply
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x

class SimpleDiT(nn.Module):
    def __init__(self, latent_dim=768, seq_len=16, hidden_dim=512, num_layers=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(num_layers)])
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaLN_modulation_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x, t):
        B = x.shape[0]
        x = self.input_proj(x)
        x = x + self.pos_embed
        t_emb = self.time_embed(t)
        for block in self.blocks:
            x = block(x, t_emb)
        shift, scale = self.adaLN_modulation_final(t_emb).chunk(2, dim=-1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.output_proj(x)
        return x

class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_schedule='linear', device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        elif beta_schedule == 'cosine':
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    @torch.no_grad()
    def p_sample(self, model, x, t):
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1)
        model_output = model(x, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    @torch.no_grad()
    def sample(self, model, shape, device):
        b = shape[0]
        x = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x
def generate_gaussian_splats(dit_model, vae_model, diffusion, num_samples=1, num_chunks=35, latent_dim=768, device='cuda', num_samples_per_cell=2):
    dit_model.eval()
    vae_model.eval()
    print(f"Generating {num_samples} Gaussian Splat samples...")
    with torch.no_grad():
        sampled_latents = diffusion.sample(dit_model, shape=(num_samples, num_chunks, latent_dim), device=device)
        print(f"Sampled latents shape: {sampled_latents.shape}")
    all_generated = []
    for i in range(num_samples):
        print(f"\nDecoding sample {i+1}/{num_samples}...")
        latent = sampled_latents[i]
        mu = latent.unsqueeze(0)
        logvar = torch.zeros_like(mu)
        all_xyz = []
        all_sh_dc = []
        all_opacity = []
        all_scale = []
        all_rotation = []
        with torch.no_grad():
            for chunk_idx in range(num_chunks):
                mu_chunk = mu[:, chunk_idx:chunk_idx+1, :]
                logvar_chunk = logvar[:, chunk_idx:chunk_idx+1, :]
                z = vae_model.reparameterize(mu_chunk.squeeze(1), logvar_chunk.squeeze(1))
                grid = vae_model.decode(z)
                recon_xyz, recon_sh_dc, recon_opacity, recon_scale, recon_rotation = \
                    vae_model.grid_to_gaussians(grid, num_samples_per_cell=num_samples_per_cell)
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
        print(f"Generated {recon_xyz.shape[1]:,} splats")
        all_generated.append({
            'xyz': recon_xyz[0],
            'sh_dc': recon_sh_dc[0],
            'opacity': recon_opacity[0],
            'scale': recon_scale[0],
            'rotation': recon_rotation[0]
        })
    return all_generated
def train_dit(dit_model, vae_model, ply_paths, num_epochs=1000, batch_size=4, lr=1e-4, device='cuda', save_path='dit_checkpoint.pt', chunk_size=4096):
    from GaussianVAE import load_3dgs_ply
    dit_model = dit_model.to(device)
    vae_model = vae_model.to(device)
    vae_model.eval()
    diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule='cosine', device=device)
    optimizer = torch.optim.AdamW(dit_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    print("Encoding all training data to latents...")
    all_latents = []
    for ply_path in tqdm(ply_paths, desc="Encoding PLY files"):
        xyz, sh_dc, opacity, scale, rotation = load_3dgs_ply(ply_path)
        N_full = len(xyz)
        mus = []
        with torch.no_grad():
            for i in range(0, N_full, chunk_size):
                end = min(i + chunk_size, N_full)
                xyz_chunk = torch.from_numpy(xyz[i:end]).unsqueeze(0).to(device)
                sh_dc_chunk = torch.from_numpy(sh_dc[i:end]).unsqueeze(0).to(device)
                opacity_chunk = torch.from_numpy(opacity[i:end]).unsqueeze(0).unsqueeze(-1).to(device)
                scale_chunk = torch.from_numpy(scale[i:end]).unsqueeze(0).to(device)
                rotation_chunk = torch.from_numpy(rotation[i:end]).unsqueeze(0).to(device)
                mu, _ = vae_model.encode(xyz_chunk, sh_dc_chunk, opacity_chunk, scale_chunk, rotation_chunk)
                mus.append(mu.squeeze(0).cpu())
        latent = torch.stack(mus, dim=0)
        all_latents.append(latent)
    all_latents = torch.stack(all_latents, dim=0)
    print(f"Encoded latents shape: {all_latents.shape}")
    best_loss = float('inf')
    for epoch in range(num_epochs):
        dit_model.train()
        epoch_loss = 0
        num_batches = 0
        perm = torch.randperm(all_latents.shape[0])
        shuffled_latents = all_latents[perm]
        for i in range(0, len(shuffled_latents), batch_size):
            batch_latents = shuffled_latents[i:i+batch_size].to(device)
            t = torch.randint(0, diffusion.num_timesteps, (batch_latents.shape[0],), device=device).long()
            noise = torch.randn_like(batch_latents)
            x_noisy = diffusion.q_sample(batch_latents, t, noise)
            noise_pred = dit_model(x_noisy, t)
            loss = nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dit_model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg_loss = epoch_loss / num_batches
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save({'epoch': epoch, 'model_state_dict': dit_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': best_loss}, save_path)


    return dit_model, diffusion

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = GaussianSplatVAE(grid_h=80, grid_w=80, latent_dim=768, hidden_dim=768)
    vae.load_state_dict(torch.load('gs_vae_cactus.pt', map_location=device)['model_state_dict'])
    vae.to(device)
    vae.eval()
    chunk_size = 4096
    num_chunks = 35
    dit = SimpleDiT(latent_dim=768, seq_len=num_chunks, hidden_dim=1024, num_layers=12, num_heads=16)
    # dit.load_state_dict(torch.load('dit_gaussplat.pt', map_location=device)['model_state_dict'])
    # dit.to(device)
    # dit.eval()
    ply_path = "/home/lu/Documents/117-final-proj/3DGS_PLY_sample_data/PLY(postshot)/cactus_splat3_30kSteps_142k_splats.ply"

    ply_paths = [ply_path]
    dit, diffusion = train_dit(dit, vae, ply_paths, num_epochs=2000, batch_size=1, lr=1e-4, device=device, save_path='dit_gaussplat.pt')

    diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule='cosine', device=device)
    generated_splats = generate_gaussian_splats(
        dit, 
        vae, 
        diffusion, 
        num_samples=4,
        num_chunks=num_chunks,
        latent_dim=768,
        device=device,
        num_samples_per_cell=2
    )
    for i, splat in enumerate(generated_splats):
        output_path = f"generated_splat_{i+1}.ply"
        save_ply(
            output_path,
            splat['xyz'],
            splat['sh_dc'],
            splat['opacity'],
            splat['scale'],
            splat['rotation']
        )
        print(f"Saved: {output_path}")