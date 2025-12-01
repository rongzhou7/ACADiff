"""
Train Diffusion Model
Train the U-Net denoiser with adaptive conditioning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import random

from config import Config
from utils.dataload import ADNIDataset
from models.vae import load_vae
from models.diffusion import UNet3D, GaussianDiffusion


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_clinical_text(prompt, method='embedding'):
    """
    Encode clinical text to embeddings
    For now using simple learnable embeddings
    TODO: Add GPT-4o encoding if Config.USE_GPT4O is True
    """
    # Placeholder: return random embedding for now
    # In real implementation, you would:
    # 1. Parse prompt to extract MMSE, ADAS13, CDR-SOB
    # 2. If USE_GPT4O: call GPT-4o API
    # 3. Else: use learnable embeddings
    return torch.randn(1, Config.CLINICAL_EMBED_DIM)


class ClinicalEncoder(nn.Module):
    """
    Simple learnable encoder for clinical scores
    Alternative to GPT-4o when Config.USE_GPT4O = False
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        # Encode 3 scores: MMSE, ADAS13, CDR-SOB
        self.mmse_embed = nn.Embedding(31, embed_dim // 3)  # MMSE: 0-30
        self.adas13_embed = nn.Linear(1, embed_dim // 3)
        self.cdr_embed = nn.Linear(1, embed_dim // 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, scores):
        """
        scores: dict with 'mmse', 'adas13', 'cdr_sb'
        Returns: [B, embed_dim]
        """
        mmse = scores['mmse'].long()  # [B]
        adas13 = scores['adas13'].unsqueeze(-1)  # [B, 1]
        cdr = scores['cdr_sb'].unsqueeze(-1)  # [B, 1]

        mmse_emb = self.mmse_embed(mmse)  # [B, dim/3]
        adas13_emb = self.adas13_embed(adas13)  # [B, dim/3]
        cdr_emb = self.cdr_embed(cdr)  # [B, dim/3]

        emb = torch.cat([mmse_emb, adas13_emb, cdr_emb], dim=-1)  # [B, dim]
        return self.proj(emb)


def modality_dropout(batch, dropout_prob=0.5):
    """
    Randomly drop modalities for adaptive training
    Returns:
        target_modality: which modality to generate
        available_modalities: which modalities are available as input
        z_avail: binary vector [B, 3]
    """
    modalities = Config.MODALITIES
    B = len(batch['subject_id'])

    # Randomly select target modality
    target_idx = random.randint(0, len(modalities) - 1)
    target_modality = modalities[target_idx]

    # Randomly decide 1->1 or 2->1
    if random.random() < dropout_prob:
        # 1->1: use one modality
        available_indices = [i for i in range(len(modalities)) if i != target_idx]
        available_idx = random.choice(available_indices)
        available_modalities = [modalities[available_idx]]
    else:
        # 2->1: use two modalities
        available_modalities = [m for m in modalities if m != target_modality]

    # Create z_avail binary vector
    z_avail = torch.zeros(B, 3)
    for i, m in enumerate(modalities):
        if m in available_modalities:
            z_avail[:, i] = 1.0

    return target_modality, available_modalities, z_avail


def train_diffusion(target_modality='VBM', device='cuda'):
    """
    Train diffusion model for generating one target modality
    """
    print(f"\n{'='*60}")
    print(f"Training Diffusion for {target_modality}")
    print(f"{'='*60}\n")

    # Set seed
    set_seed(Config.SEED)

    # Create directories
    Config.create_dirs()
    save_dir = os.path.join(Config.SAVE_DIR, 'diffusion', target_modality)
    os.makedirs(save_dir, exist_ok=True)

    # Load VAEs (frozen encoders)
    print("Loading VAE encoders...")
    vaes = {}
    for modality in Config.MODALITIES:
        try:
            vaes[modality] = load_vae(modality, device=device)
            vaes[modality].eval()
            for param in vaes[modality].parameters():
                param.requires_grad = False
            print(f"  {modality} VAE loaded")
        except Exception as e:
            print(f"  Warning: Could not load VAE for {modality}: {e}")
            print(f"  You need to train VAE first: python vae.py --modality {modality}")
            return

    # Create datasets
    train_dataset = ADNIDataset(split='train')
    val_dataset = ADNIDataset(split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.DIFF_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.DIFF_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    # Create diffusion model
    print("\nCreating U-Net model...")
    unet = UNet3D(
        latent_channels=Config.VAE_LATENT_CHANNELS,
        base_channels=Config.UNET_BASE_CHANNELS,
        ch_mult=Config.UNET_CH_MULT,
        num_res_blocks=Config.UNET_NUM_RES_BLOCKS,
        attn_resolutions=Config.UNET_ATTN_RESOLUTIONS,
        clinical_dim=Config.CLINICAL_EMBED_DIM
    ).to(device)

    # Clinical encoder
    clinical_encoder = ClinicalEncoder(embed_dim=Config.CLINICAL_EMBED_DIM).to(device)

    # Diffusion process
    diffusion = GaussianDiffusion(
        timesteps=Config.DIFFUSION_TIMESTEPS,
        beta_schedule=Config.DIFFUSION_BETA_SCHEDULE,
        beta_start=Config.DIFFUSION_BETA_START,
        beta_end=Config.DIFFUSION_BETA_END
    )

    # Optimizer (for both U-Net and clinical encoder)
    params = list(unet.parameters()) + list(clinical_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=Config.DIFF_LEARNING_RATE)

    # Training loop
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(Config.DIFF_NUM_EPOCHS):
        # Train
        unet.train()
        clinical_encoder.train()
        train_loss = 0
        train_noise_loss = 0
        train_cons_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.DIFF_NUM_EPOCHS}')
        for batch in pbar:
            # Move data to device
            images = {k: v.to(device) for k, v in batch['images'].items()}
            scores = {k: v.to(device) for k, v in batch['scores'].items()}

            # Encode to latent with VAEs (no gradient)
            with torch.no_grad():
                latents = {}
                for modality in Config.MODALITIES:
                    mean, logvar = vaes[modality].encode(images[modality])
                    # Use mean for training (no sampling)
                    latents[modality] = mean

            # Modality dropout to determine target and available modalities
            # For simplicity, we fix target_modality as the one we're training
            # and randomly select 1 or 2 available modalities
            available_modalities = [m for m in Config.MODALITIES if m != target_modality]
            if random.random() < Config.MODALITY_DROPOUT_PROB:
                # 1->1
                available_modalities = [random.choice(available_modalities)]

            # Create z_avail
            B = len(batch['subject_id'])
            z_avail = torch.zeros(B, 3).to(device)
            for i, m in enumerate(Config.MODALITIES):
                if m in available_modalities:
                    z_avail[:, i] = 1.0

            # Get target latent
            z_0 = latents[target_modality]

            # Get available latents
            z_avail_list = [latents[m] for m in available_modalities]

            # Encode clinical data
            # Random clinical dropout
            if random.random() < Config.CLINICAL_DROPOUT_PROB:
                text_emb = torch.zeros(B, Config.CLINICAL_EMBED_DIM).to(device)
            else:
                text_emb = clinical_encoder(scores)

            # Sample timestep
            t = torch.randint(0, Config.DIFFUSION_TIMESTEPS, (B,), device=device)

            # Forward diffusion: add noise
            noise = torch.randn_like(z_0)
            z_t = diffusion.q_sample(z_0, t, noise=noise)

            # Predict noise
            noise_pred = unet(z_t, z_avail_list, z_avail, text_emb, t)

            # Loss: noise prediction
            loss_noise = F.mse_loss(noise_pred, noise)

            # Consistency loss: predict z_0 and compare
            z_0_pred = diffusion.predict_start_from_noise(z_t, t, noise_pred)
            loss_cons = F.mse_loss(z_0_pred, z_0)

            # Total loss
            loss = loss_noise + Config.DIFF_CONSISTENCY_WEIGHT * loss_cons

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, Config.DIFF_GRAD_CLIP)
            optimizer.step()

            # Stats
            train_loss += loss.item()
            train_noise_loss += loss_noise.item()
            train_cons_loss += loss_cons.item()
            global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'noise': f'{loss_noise.item():.4f}',
                'cons': f'{loss_cons.item():.4f}'
            })

        train_loss /= len(train_loader)
        train_noise_loss /= len(train_loader)
        train_cons_loss /= len(train_loader)

        # Validation
        unet.eval()
        clinical_encoder.eval()
        val_loss = 0
        val_noise_loss = 0
        val_cons_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                images = {k: v.to(device) for k, v in batch['images'].items()}
                scores = {k: v.to(device) for k, v in batch['scores'].items()}

                # Encode to latent
                latents = {}
                for modality in Config.MODALITIES:
                    mean, logvar = vaes[modality].encode(images[modality])
                    latents[modality] = mean

                # Use all available modalities for validation
                available_modalities = [m for m in Config.MODALITIES if m != target_modality]
                B = len(batch['subject_id'])
                z_avail = torch.zeros(B, 3).to(device)
                for i, m in enumerate(Config.MODALITIES):
                    if m in available_modalities:
                        z_avail[:, i] = 1.0

                z_0 = latents[target_modality]
                z_avail_list = [latents[m] for m in available_modalities]
                text_emb = clinical_encoder(scores)

                t = torch.randint(0, Config.DIFFUSION_TIMESTEPS, (B,), device=device)
                noise = torch.randn_like(z_0)
                z_t = diffusion.q_sample(z_0, t, noise=noise)

                noise_pred = unet(z_t, z_avail_list, z_avail, text_emb, t)
                loss_noise = F.mse_loss(noise_pred, noise)

                z_0_pred = diffusion.predict_start_from_noise(z_t, t, noise_pred)
                loss_cons = F.mse_loss(z_0_pred, z_0)

                loss = loss_noise + Config.DIFF_CONSISTENCY_WEIGHT * loss_cons

                val_loss += loss.item()
                val_noise_loss += loss_noise.item()
                val_cons_loss += loss_cons.item()

        val_loss /= len(val_loader)
        val_noise_loss /= len(val_loader)
        val_cons_loss /= len(val_loader)

        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{Config.DIFF_NUM_EPOCHS}:')
        print(f'  Train - Loss: {train_loss:.4f}, Noise: {train_noise_loss:.4f}, Cons: {train_cons_loss:.4f}')
        print(f'  Val   - Loss: {val_loss:.4f}, Noise: {val_noise_loss:.4f}, Cons: {val_cons_loss:.4f}')

        # Save checkpoint
        if (epoch + 1) % Config.SAVE_INTERVAL == 0:
            checkpoint = {
                'epoch': epoch,
                'unet_state_dict': unet.state_dict(),
                'clinical_encoder_state_dict': clinical_encoder.state_dict(),
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
                'unet_state_dict': unet.state_dict(),
                'clinical_encoder_state_dict': clinical_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            print(f'  -> Best model saved! (val_loss: {val_loss:.4f})')

    print(f"\nTraining completed for {target_modality}!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Diffusion Model')
    parser.add_argument('--target', type=str, default='VBM',
                        choices=['VBM', 'FDG', 'AV45'],
                        help='Target modality to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Check config
    print("Current Configuration:")
    print(f"  VAE Latent Channels: {Config.VAE_LATENT_CHANNELS}")
    print(f"  U-Net Base Channels: {Config.UNET_BASE_CHANNELS}")
    print(f"  Diffusion Timesteps: {Config.DIFFUSION_TIMESTEPS}")
    print(f"  Batch Size: {Config.DIFF_BATCH_SIZE}")
    print(f"  Learning Rate: {Config.DIFF_LEARNING_RATE}")
    print(f"  Num Epochs: {Config.DIFF_NUM_EPOCHS}")
    print(f"  Consistency Weight: {Config.DIFF_CONSISTENCY_WEIGHT}")
    print(f"  Modality Dropout Prob: {Config.MODALITY_DROPOUT_PROB}")
    print(f"  Clinical Dropout Prob: {Config.CLINICAL_DROPOUT_PROB}")

    # Train
    train_diffusion(target_modality=args.target, device=args.device)
