"""
Diffusion Model with Adaptive Conditioning
U-Net denoiser + DDPM/DDIM + hierarchical conditioning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from config import Config


# ========== Helper Functions ==========

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ========== U-Net Building Blocks ==========

class ResBlock3D(nn.Module):
    """3D Residual Block with GroupNorm and timestep modulation (FiLM)"""
    def __init__(self, in_channels, out_channels=None, temb_channels=512):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        # Timestep modulation (FiLM)
        self.temb_proj = nn.Linear(temb_channels, out_channels * 2)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))

        # FiLM: gamma * h + beta
        temb_out = self.temb_proj(F.silu(temb))[:, :, None, None, None]
        gamma, beta = torch.chunk(temb_out, 2, dim=1)
        h = gamma * h + beta

        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class SpatialCrossAttention3D(nn.Module):
    """3D Cross-Attention for clinical text conditioning"""
    def __init__(self, channels, context_dim=512, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.context_proj = nn.Linear(context_dim, channels * 2)
        self.proj_out = nn.Conv3d(channels, channels, 1)

    def forward(self, x, context):
        """
        x: [B, C, D, H, W]
        context: [B, context_dim]
        """
        B, C, D, H, W = x.shape

        # Normalize
        h = self.norm(x)

        # Q from image features
        qkv = self.qkv(h)
        q, k_img, v_img = torch.chunk(qkv, 3, dim=1)

        # K, V from context
        kv_context = self.context_proj(context)  # [B, C*2]
        k_context, v_context = torch.chunk(kv_context, 2, dim=1)  # [B, C]

        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, self.head_dim, D * H * W).permute(0, 1, 3, 2)  # [B, heads, DHW, head_dim]
        k = k_context.reshape(B, self.num_heads, self.head_dim, 1).permute(0, 1, 3, 2)  # [B, heads, 1, head_dim]
        v = v_context.reshape(B, self.num_heads, self.head_dim, 1).permute(0, 1, 3, 2)  # [B, heads, 1, head_dim]

        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B, heads, DHW, 1]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, DHW, head_dim]

        # Reshape back
        out = out.permute(0, 1, 3, 2).reshape(B, C, D, H, W)
        out = self.proj_out(out)

        return x + out


class Downsample3D(nn.Module):
    """Downsample by 2x"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """Upsample by 2x"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ========== Adaptive Fusion Module ==========

class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion of available modalities
    - If 2 modalities available: Cross-Attention fusion
    - If 1 modality available: Projection
    """
    def __init__(self, latent_channels=8, num_heads=4):
        super().__init__()

        # For 2->1: Cross-attention between two modalities
        self.cross_attn = SpatialCrossAttention3D(latent_channels, context_dim=latent_channels, num_heads=num_heads)

        # For 1->1: Simple projection
        self.proj = nn.Conv3d(latent_channels, latent_channels, 3, padding=1)

    def forward(self, z_list, z_avail):
        """
        z_list: list of available latent codes, each [B, C, D, H, W]
        z_avail: [B, 3] binary vector indicating availability
        Returns: fused features [B, C, D, H, W]
        """
        num_available = z_avail.sum(dim=1)[0].item()  # Assume same availability in batch

        if num_available == 2:
            # Cross-attention fusion
            z_i, z_j = z_list[0], z_list[1]
            # Simple pooling for context
            z_j_pooled = z_j.mean(dim=[2, 3, 4])  # [B, C]
            fused = self.cross_attn(z_i, z_j_pooled)
            return fused

        elif num_available == 1:
            # Projection
            z_i = z_list[0]
            return self.proj(z_i)

        else:
            raise ValueError(f"Unexpected number of available modalities: {num_available}")


# ========== U-Net Denoiser ==========

class UNet3D(nn.Module):
    """
    3D U-Net denoiser with adaptive conditioning
    """
    def __init__(self,
                 latent_channels=8,
                 base_channels=128,
                 ch_mult=[1, 2, 4, 8],
                 num_res_blocks=2,
                 attn_resolutions=[16],
                 clinical_dim=512):
        super().__init__()

        self.latent_channels = latent_channels
        self.base_channels = base_channels

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        temb_channels = base_channels * 4

        # Adaptive fusion for image conditioning
        self.adaptive_fusion = AdaptiveFusion(latent_channels)

        # Input projection (noisy target + fused conditioning)
        self.conv_in = nn.Conv3d(latent_channels * 2, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        channels = [base_channels * m for m in ch_mult]
        in_ch = base_channels

        for i, out_ch in enumerate(channels):
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(in_ch, out_ch, temb_channels))
                in_ch = out_ch
            if i < len(channels) - 1:
                block.append(Downsample3D(out_ch))
            self.down_blocks.append(block)

        # Middle
        self.mid_block1 = ResBlock3D(channels[-1], channels[-1], temb_channels)
        self.mid_attn = SpatialCrossAttention3D(channels[-1], clinical_dim)
        self.mid_block2 = ResBlock3D(channels[-1], channels[-1], temb_channels)

        # Decoder
        self.up_blocks = nn.ModuleList()
        channels_reversed = list(reversed(channels))

        for i, out_ch in enumerate(channels_reversed):
            block = nn.ModuleList()
            in_ch = out_ch
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(in_ch, out_ch, temb_channels))
            if i < len(channels_reversed) - 1:
                block.append(Upsample3D(out_ch))
            self.up_blocks.append(block)

        # Output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=base_channels)
        self.conv_out = nn.Conv3d(base_channels, latent_channels, 3, padding=1)

    def forward(self, z_t, z_avail_list, z_avail, text_emb, t):
        """
        z_t: noisy target latent [B, C, D, H, W]
        z_avail_list: list of available modality latents
        z_avail: binary vector [B, 3]
        text_emb: clinical text embedding [B, clinical_dim]
        t: timestep [B]
        """
        # Timestep embedding
        temb = self.time_embed(timestep_embedding(t, self.base_channels))

        # Adaptive image fusion
        z_fused = self.adaptive_fusion(z_avail_list, z_avail)

        # Concatenate noisy target with fused conditioning
        h = torch.cat([z_t, z_fused], dim=1)
        h = self.conv_in(h)

        # Encoder
        for block_list in self.down_blocks:
            for block in block_list:
                if isinstance(block, ResBlock3D):
                    h = block(h, temb)
                else:
                    h = block(h)

        # Middle with clinical cross-attention
        h = self.mid_block1(h, temb)
        h = self.mid_attn(h, text_emb)
        h = self.mid_block2(h, temb)

        # Decoder
        for block_list in self.up_blocks:
            for block in block_list:
                if isinstance(block, ResBlock3D):
                    h = block(h, temb)
                else:
                    h = block(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


# ========== Diffusion Process ==========

class GaussianDiffusion:
    """
    DDPM with cosine noise schedule
    """
    def __init__(self,
                 timesteps=1000,
                 beta_schedule='cosine',
                 beta_start=0.0001,
                 beta_end=0.02):
        self.timesteps = timesteps

        # Create noise schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'cosine':
            betas = self.cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in Improved DDPM"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, z_0, t, noise=None):
        """
        Forward diffusion: add noise to z_0
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(z_0.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(z_0.device)

        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(z_0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_alphas_cumprod_t * z_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, z_t, t, noise):
        """
        Predict z_0 from z_t and predicted noise
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(z_t.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(z_t.device)

        while len(sqrt_alphas_cumprod_t.shape) < len(z_t.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return (z_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t


if __name__ == '__main__':
    # Test U-Net
    print("Testing U-Net...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet3D(
        latent_channels=Config.VAE_LATENT_CHANNELS,
        base_channels=Config.UNET_BASE_CHANNELS,
        ch_mult=Config.UNET_CH_MULT,
        num_res_blocks=Config.UNET_NUM_RES_BLOCKS
    ).to(device)

    # Dummy inputs
    B = 2
    z_t = torch.randn(B, Config.VAE_LATENT_CHANNELS, 20, 22, 20).to(device)
    z_avail_list = [
        torch.randn(B, Config.VAE_LATENT_CHANNELS, 20, 22, 20).to(device),
        torch.randn(B, Config.VAE_LATENT_CHANNELS, 20, 22, 20).to(device)
    ]
    z_avail = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32).to(device)
    text_emb = torch.randn(B, Config.CLINICAL_EMBED_DIM).to(device)
    t = torch.randint(0, Config.DIFFUSION_TIMESTEPS, (B,)).to(device)

    # Forward
    noise_pred = model(z_t, z_avail_list, z_avail, text_emb, t)
    print(f"Input shape: {z_t.shape}")
    print(f"Output shape: {noise_pred.shape}")
    print("U-Net test passed!")

    # Test diffusion
    print("\nTesting Diffusion...")
    diffusion = GaussianDiffusion(
        timesteps=Config.DIFFUSION_TIMESTEPS,
        beta_schedule=Config.DIFFUSION_BETA_SCHEDULE
    )

    z_0 = torch.randn(B, Config.VAE_LATENT_CHANNELS, 20, 22, 20).to(device)
    t = torch.randint(0, Config.DIFFUSION_TIMESTEPS, (B,))
    z_t = diffusion.q_sample(z_0, t)
    print(f"z_0 shape: {z_0.shape}")
    print(f"z_t shape: {z_t.shape}")
    print("Diffusion test passed!")
