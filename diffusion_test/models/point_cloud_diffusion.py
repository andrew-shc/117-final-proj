"""
Point Cloud Diffusion Model Implementation
Based on DDPM (Denoising Diffusion Probabilistic Models) for 3D point clouds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PointNet(nn.Module):
    """
    PointNet-based backbone for point cloud feature extraction
    """
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # MLP layers for point-wise features
        self.conv1 = nn.Conv1d(input_dim + 1, 64, 1)  # +1 for time embedding
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, hidden_dim, 1)

        # MLP for output
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

    def forward(self, x, t):
        """
        Args:
            x: (B, N, 3) point cloud
            t: (B,) time steps
        Returns:
            (B, N, 3) denoised point cloud
        """
        B, N, C = x.shape

        # Time embedding
        t_emb = self.get_time_embedding(t, B, N)  # (B, N, 1)

        # Concatenate time embedding
        x = torch.cat([x, t_emb], dim=-1)  # (B, N, 4)
        x = x.transpose(1, 2)  # (B, 4, N)

        # Point-wise features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Per-point features
        x = x.transpose(1, 2)  # (B, N, hidden_dim)

        # MLP
        x = F.relu(self.bn4(self.fc1(x).transpose(1, 2)).transpose(1, 2))
        x = F.relu(self.bn5(self.fc2(x).transpose(1, 2)).transpose(1, 2))
        x = self.fc3(x)

        return x

    def get_time_embedding(self, t, batch_size, num_points):
        """Generate time embeddings"""
        t = t.view(batch_size, 1, 1)
        t = t.expand(batch_size, num_points, 1)
        return t.float()


class PointCloudDiffusionModel(nn.Module):
    """
    Diffusion Model for 3D Point Clouds
    """
    def __init__(
        self,
        num_points: int = 1024,
        point_dim: int = 3,
        hidden_dim: int = 128,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_points = num_points
        self.point_dim = point_dim
        self.num_timesteps = num_timesteps

        # Denoising network
        self.denoiser = PointNet(
            input_dim=point_dim,
            hidden_dim=hidden_dim,
            output_dim=point_dim
        )

        # Diffusion schedule
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps, beta_start, beta_end))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev',
                           F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))

        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                           torch.sqrt(1.0 - self.alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance',
                           self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        """Cosine schedule as proposed in DDPM"""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Forward diffusion process: q(x_t | x_0)
        Args:
            x_0: (B, N, 3) original point cloud
            t: (B,) time steps
            noise: (B, N, 3) optional noise
        Returns:
            Noisy point cloud at time t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Training loss: predict the noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        x_noisy = self.q_sample(x_0, t, noise)
        predicted_noise = self.denoiser(x_noisy, t)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int):
        """
        Reverse diffusion process: sample x_{t-1} from x_t
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=x_t.device)

        # Predict noise
        predicted_noise = self.denoiser(x_t, t_tensor)

        # Calculate x_0 estimate
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod_prev[t]
        beta_t = self.betas[t]

        # Calculate mean
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

        # Clip x_0 prediction
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Calculate mean of p(x_{t-1} | x_t)
        mean = (torch.sqrt(alpha_t_prev) * beta_t / (1 - alpha_t)) * x_0_pred + \
               (torch.sqrt(1 - beta_t) * (1 - alpha_t_prev) / (1 - alpha_t)) * x_t

        if t == 0:
            return mean
        else:
            variance = self.posterior_variance[t]
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, batch_size: int = 1, device: str = 'cpu'):
        """
        Generate point clouds from noise
        """
        shape = (batch_size, self.num_points, self.point_dim)
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t)

        return x

    @torch.no_grad()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, t: int = 500, lambda_: float = 0.5):
        """
        Interpolate between two point clouds in latent space
        """
        batch_size = x1.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=x1.device)

        # Add noise to both
        noise1 = torch.randn_like(x1)
        noise2 = torch.randn_like(x2)

        x1_t = self.q_sample(x1, t_tensor, noise1)
        x2_t = self.q_sample(x2, t_tensor, noise2)

        # Interpolate in noisy space
        x_t = (1 - lambda_) * x1_t + lambda_ * x2_t

        # Denoise
        for t_step in reversed(range(t + 1)):
            x_t = self.p_sample(x_t, t_step)

        return x_t

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        Training forward pass
        """
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)

        return self.p_losses(x, t)
