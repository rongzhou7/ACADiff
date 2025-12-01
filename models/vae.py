"""
3D Variational Autoencoder with KL Divergence
Compress 3D brain images to latent space
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.dataload import ADNIDataset


class ResBlock3D(nn.Module):
    """3D Residual Block with GroupNorm"""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class Downsample3D(nn.Module):
    """Downsample by 2x using conv stride"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """Upsample by 2x using interpolation + conv"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Encoder3D(nn.Module):
    """
    3D Encoder
    Input: [B, 1, 160, 180, 160]
    Output: [B, latent_channels*2, 20, 22, 20] (mean and logvar)
    """
    def __init__(self,
                 in_channels=1,
                 base_channels=64,
                 latent_channels=8,
                 ch_mult=[1, 2, 4, 8],
                 num_res_blocks=2):
        super().__init__()

        # Initial conv
        self.conv_in = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        channels = [base_channels * m for m in ch_mult]
        in_ch = base_channels

        for i, out_ch in enumerate(channels):
            block = nn.ModuleList()
            # Residual blocks
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(in_ch, out_ch))
                in_ch = out_ch
            # Downsample (except last layer)
            if i < len(channels) - 1:
                block.append(Downsample3D(out_ch))
            self.down_blocks.append(block)

        # Middle block
        self.mid_block1 = ResBlock3D(channels[-1])
        self.mid_block2 = ResBlock3D(channels[-1])

        # Output to latent
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=channels[-1])
        self.conv_out = nn.Conv3d(channels[-1], latent_channels * 2, 3, padding=1)

    def forward(self, x):
        # Initial conv
        h = self.conv_in(x)

        # Downsample
        for block_list in self.down_blocks:
            for block in block_list:
                h = block(h)

        # Middle
        h = self.mid_block1(h)
        h = self.mid_block2(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class Decoder3D(nn.Module):
    """
    3D Decoder
    Input: [B, latent_channels, 20, 22, 20]
    Output: [B, 1, 160, 180, 160]
    """
    def __init__(self,
                 out_channels=1,
                 base_channels=64,
                 latent_channels=8,
                 ch_mult=[1, 2, 4, 8],
                 num_res_blocks=2):
        super().__init__()

        channels = [base_channels * m for m in ch_mult]

        # Input from latent
        self.conv_in = nn.Conv3d(latent_channels, channels[-1], 3, padding=1)

        # Middle block
        self.mid_block1 = ResBlock3D(channels[-1])
        self.mid_block2 = ResBlock3D(channels[-1])

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        channels_reversed = list(reversed(channels))

        for i, out_ch in enumerate(channels_reversed):
            block = nn.ModuleList()
            in_ch = out_ch
            # Residual blocks
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(in_ch))
            # Upsample (except last layer)
            if i < len(channels_reversed) - 1:
                block.append(Upsample3D(in_ch))
            self.up_blocks.append(block)

        # Output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=base_channels)
        self.conv_out = nn.Conv3d(base_channels, out_channels, 3, padding=1)

    def forward(self, z):
        # Input
        h = self.conv_in(z)

        # Middle
        h = self.mid_block1(h)
        h = self.mid_block2(h)

        # Upsample
        for block_list in self.up_blocks:
            for block in block_list:
                h = block(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class VAE3D(nn.Module):
    """
    3D Variational Autoencoder with KL divergence
    """
    def __init__(self,
                 in_channels=1,
                 base_channels=64,
                 latent_channels=8,
                 ch_mult=[1, 2, 4, 8],
                 num_res_blocks=2):
        super().__init__()

        self.encoder = Encoder3D(in_channels, base_channels, latent_channels, ch_mult, num_res_blocks)
        self.decoder = Decoder3D(in_channels, base_channels, latent_channels, ch_mult, num_res_blocks)
        self.latent_channels = latent_channels

    def encode(self, x):
        """Encode to latent distribution parameters"""
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        """Decode from latent"""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar


def vae_loss(recon, x, mean, logvar, kl_weight=1e-6):
    """
    VAE loss = Reconstruction + KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, x, reduction='mean')

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    # Total loss
    loss = recon_loss + kl_weight * kl_loss

    return loss, recon_loss, kl_loss


def train_vae(modality='VBM', device='cuda'):
    """
    Train VAE for one modality
    """
    print(f"\n{'='*60}")
    print(f"Training VAE for {modality}")
    print(f"{'='*60}\n")

    # Create directories
    Config.create_dirs()
    save_dir = os.path.join(Config.SAVE_DIR, 'vae', modality)
    os.makedirs(save_dir, exist_ok=True)

    # Create dataset
    train_dataset = ADNIDataset(
        data_root=Config.DATA_ROOT,
        csv_path=Config.CSV_PATH,
        modalities=Config.MODALITIES,
        target_shape=Config.TARGET_SHAPE,
        split='train',
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO
    )

    val_dataset = ADNIDataset(
        data_root=Config.DATA_ROOT,
        csv_path=Config.CSV_PATH,
        modalities=Config.MODALITIES,
        target_shape=Config.TARGET_SHAPE,
        split='val',
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.VAE_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAE_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    # Create model
    model = VAE3D(
        in_channels=1,
        base_channels=Config.VAE_BASE_CHANNELS,
        latent_channels=Config.VAE_LATENT_CHANNELS,
        ch_mult=Config.VAE_CH_MULT,
        num_res_blocks=Config.VAE_NUM_RES_BLOCKS
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.VAE_LEARNING_RATE)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(Config.VAE_NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.VAE_NUM_EPOCHS}')
        for batch in pbar:
            # Get data for this modality
            x = batch['images'][modality].to(device)

            # Forward
            recon, mean, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mean, logvar, Config.VAE_KL_WEIGHT)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.VAE_GRAD_CLIP)
            optimizer.step()

            # Stats
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.6f}'
            })

        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch['images'][modality].to(device)
                recon, mean, logvar = model(x)
                loss, recon_loss, kl_loss = vae_loss(recon, x, mean, logvar, Config.VAE_KL_WEIGHT)

                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()

        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_kl_loss /= len(val_loader)

        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{Config.VAE_NUM_EPOCHS}:')
        print(f'  Train - Loss: {train_loss:.4f}, Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.6f}')
        print(f'  Val   - Loss: {val_loss:.4f}, Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.6f}')

        # Save checkpoint
        if (epoch + 1) % Config.SAVE_INTERVAL == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pt'))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            print(f'  -> Best model saved! (val_loss: {val_loss:.4f})')

    print(f"\nTraining completed for {modality}!")
    print(f"Best validation loss: {best_val_loss:.4f}")


def load_vae(modality, device='cuda'):
    """
    Load trained VAE model
    """
    model = VAE3D(
        in_channels=1,
        base_channels=Config.VAE_BASE_CHANNELS,
        latent_channels=Config.VAE_LATENT_CHANNELS,
        ch_mult=Config.VAE_CH_MULT,
        num_res_blocks=Config.VAE_NUM_RES_BLOCKS
    ).to(device)

    checkpoint_path = os.path.join(Config.SAVE_DIR, 'vae', modality, 'best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train 3D VAE')
    parser.add_argument('--modality', type=str, default='VBM',
                        choices=['VBM', 'FDG', 'AV45'],
                        help='Which modality to train')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Train VAE
    train_vae(modality=args.modality, device=args.device)
