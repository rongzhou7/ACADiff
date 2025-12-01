"""
Inference: Generate Missing Modalities
DDIM sampling to generate target modality from available ones
"""
import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import nibabel as nib

from config import Config
from utils.dataload import ADNIDataset
from models.vae import load_vae
from models.diffusion import UNet3D, GaussianDiffusion
from train_diffusion import ClinicalEncoder


def load_diffusion_model(target_modality, device='cuda'):
    """
    Load trained diffusion model for target modality
    """
    # Load U-Net
    unet = UNet3D(
        latent_channels=Config.VAE_LATENT_CHANNELS,
        base_channels=Config.UNET_BASE_CHANNELS,
        ch_mult=Config.UNET_CH_MULT,
        num_res_blocks=Config.UNET_NUM_RES_BLOCKS,
        attn_resolutions=Config.UNET_ATTN_RESOLUTIONS,
        clinical_dim=Config.CLINICAL_EMBED_DIM
    ).to(device)

    # Load clinical encoder
    clinical_encoder = ClinicalEncoder(embed_dim=Config.CLINICAL_EMBED_DIM).to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(Config.SAVE_DIR, 'diffusion', target_modality, 'best_model.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Train the model first!")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    clinical_encoder.load_state_dict(checkpoint['clinical_encoder_state_dict'])

    unet.eval()
    clinical_encoder.eval()

    print(f"Loaded diffusion model for {target_modality} (epoch {checkpoint['epoch']})")

    return unet, clinical_encoder


class DDIMSampler:
    """
    DDIM sampling for faster inference
    """
    def __init__(self, diffusion, num_inference_steps=50):
        self.diffusion = diffusion
        self.num_inference_steps = num_inference_steps

        # Create subset of timesteps for DDIM
        step_ratio = diffusion.timesteps // num_inference_steps
        self.timesteps = torch.arange(0, diffusion.timesteps, step_ratio).long()
        self.timesteps = torch.flip(self.timesteps, [0])  # Reverse: T -> 0

    @torch.no_grad()
    def sample(self, unet, z_avail_list, z_avail, text_emb, shape, device='cuda'):
        """
        DDIM sampling
        Args:
            unet: denoiser model
            z_avail_list: list of available modality latents
            z_avail: binary availability vector
            text_emb: clinical text embedding
            shape: target latent shape [B, C, D, H, W]
        Returns:
            z_0: generated latent
        """
        # Start from pure noise
        z_t = torch.randn(shape, device=device)

        for i, t in enumerate(tqdm(self.timesteps, desc='DDIM Sampling')):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = unet(z_t, z_avail_list, z_avail, text_emb, t_batch)

            # Predict z_0
            z_0_pred = self.diffusion.predict_start_from_noise(z_t, t_batch, noise_pred)

            # DDIM update (deterministic)
            if i < len(self.timesteps) - 1:
                t_prev = self.timesteps[i + 1]
                alpha_t = self.diffusion.alphas_cumprod[t].to(device)
                alpha_t_prev = self.diffusion.alphas_cumprod[t_prev].to(device)

                # Reshape for broadcasting
                while len(alpha_t.shape) < len(z_t.shape):
                    alpha_t = alpha_t.unsqueeze(-1)
                    alpha_t_prev = alpha_t_prev.unsqueeze(-1)

                # DDIM formula
                sigma_t = 0  # eta = 0 for deterministic
                dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * noise_pred
                z_t = torch.sqrt(alpha_t_prev) * z_0_pred + dir_xt
            else:
                z_t = z_0_pred

        return z_t


def generate_missing_modality(
    target_modality,
    available_modalities,
    batch,
    vaes,
    unet,
    clinical_encoder,
    sampler,
    device='cuda',
    num_samples=1
):
    """
    Generate missing target modality from available modalities

    Args:
        target_modality: which modality to generate (e.g., 'VBM')
        available_modalities: list of available modalities (e.g., ['FDG', 'AV45'])
        batch: data batch from dataloader
        vaes: dict of VAE models
        unet: U-Net denoiser
        clinical_encoder: clinical data encoder
        sampler: DDIM sampler
        device: cuda or cpu
        num_samples: number of samples for Monte Carlo (paper uses 10)

    Returns:
        generated_images: [num_samples, B, 1, D, H, W]
    """
    B = len(batch['subject_id'])

    # Move data to device
    images = {k: v.to(device) for k, v in batch['images'].items()}
    scores = {k: v.to(device) for k, v in batch['scores'].items()}

    # Encode available modalities to latent
    z_avail_list = []
    for modality in available_modalities:
        with torch.no_grad():
            mean, logvar = vaes[modality].encode(images[modality])
            z_avail_list.append(mean)

    # Create z_avail binary vector
    z_avail = torch.zeros(B, 3).to(device)
    for i, m in enumerate(Config.MODALITIES):
        if m in available_modalities:
            z_avail[:, i] = 1.0

    # Encode clinical data
    with torch.no_grad():
        text_emb = clinical_encoder(scores)

    # Generate multiple samples (Monte Carlo)
    generated_latents = []
    for _ in range(num_samples):
        z_0 = sampler.sample(
            unet,
            z_avail_list,
            z_avail,
            text_emb,
            shape=(B, Config.VAE_LATENT_CHANNELS, *Config.VAE_LATENT_SHAPE),
            device=device
        )
        generated_latents.append(z_0)

    # Decode latents to images
    generated_images = []
    for z_0 in generated_latents:
        with torch.no_grad():
            img = vaes[target_modality].decode(z_0)
            generated_images.append(img)

    # Stack: [num_samples, B, 1, D, H, W]
    generated_images = torch.stack(generated_images, dim=0)

    return generated_images


def inference_pipeline(
    target_modality='VBM',
    available_modalities=['FDG', 'AV45'],
    split='test',
    num_samples=10,
    save_results=True,
    device='cuda'
):
    """
    Full inference pipeline
    Generate missing modality for entire dataset
    """
    print(f"\n{'='*60}")
    print(f"Inference: Generate {target_modality} from {available_modalities}")
    print(f"{'='*60}\n")

    # Validate input
    if target_modality in available_modalities:
        raise ValueError(f"Target modality {target_modality} should not be in available modalities!")

    # Load VAEs
    print("Loading VAE models...")
    vaes = {}
    for modality in Config.MODALITIES:
        vaes[modality] = load_vae(modality, device=device)
        vaes[modality].eval()

    # Load diffusion model
    print("Loading diffusion model...")
    unet, clinical_encoder = load_diffusion_model(target_modality, device=device)

    # Create diffusion and sampler
    diffusion = GaussianDiffusion(
        timesteps=Config.DIFFUSION_TIMESTEPS,
        beta_schedule=Config.DIFFUSION_BETA_SCHEDULE,
        beta_start=Config.DIFFUSION_BETA_START,
        beta_end=Config.DIFFUSION_BETA_END
    )

    sampler = DDIMSampler(diffusion, num_inference_steps=Config.INFERENCE_NUM_STEPS)

    # Create dataset
    print(f"Loading {split} dataset...")
    dataset = ADNIDataset(split=split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Config.INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )

    # Inference
    print(f"\nGenerating {target_modality} for {len(dataset)} samples...")
    print(f"Monte Carlo samples: {num_samples}")

    all_generated = []
    all_ground_truth = []
    all_subject_ids = []

    for batch in tqdm(dataloader, desc='Inference'):
        # Generate
        generated = generate_missing_modality(
            target_modality=target_modality,
            available_modalities=available_modalities,
            batch=batch,
            vaes=vaes,
            unet=unet,
            clinical_encoder=clinical_encoder,
            sampler=sampler,
            device=device,
            num_samples=num_samples
        )

        # Average over Monte Carlo samples
        generated_mean = generated.mean(dim=0)  # [B, 1, D, H, W]

        all_generated.append(generated_mean.cpu())
        all_ground_truth.append(batch['images'][target_modality].cpu())
        all_subject_ids.extend(batch['subject_id'])

    # Concatenate all results
    all_generated = torch.cat(all_generated, dim=0)  # [N, 1, D, H, W]
    all_ground_truth = torch.cat(all_ground_truth, dim=0)  # [N, 1, D, H, W]

    print(f"\nGeneration completed!")
    print(f"Generated shape: {all_generated.shape}")

    # Save results
    if save_results:
        save_dir = os.path.join(Config.SAVE_DIR, 'inference', target_modality)
        os.makedirs(save_dir, exist_ok=True)

        torch.save({
            'generated': all_generated,
            'ground_truth': all_ground_truth,
            'subject_ids': all_subject_ids,
            'target_modality': target_modality,
            'available_modalities': available_modalities,
            'num_samples': num_samples
        }, os.path.join(save_dir, f'inference_{split}.pt'))

        print(f"Results saved to {save_dir}")

    return all_generated, all_ground_truth, all_subject_ids


def save_as_nifti(tensor, subject_id, modality, save_dir):
    """
    Save tensor as NIfTI file
    """
    # tensor: [1, D, H, W]
    data = tensor.squeeze(0).cpu().numpy()  # [D, H, W]

    # Denormalize from [-1, 1] to original range
    # This is a simple denormalization; adjust based on your data
    data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]

    # Create NIfTI image
    img = nib.Nifti1Image(data, affine=np.eye(4))

    # Save
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f'{subject_id}_{modality}_generated.nii.gz')
    nib.save(img, filepath)

    return filepath


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate Missing Modalities')
    parser.add_argument('--target', type=str, default='VBM',
                        choices=['VBM', 'FDG', 'AV45'],
                        help='Target modality to generate')
    parser.add_argument('--available', type=str, nargs='+', default=['FDG', 'AV45'],
                        help='Available modalities (e.g., --available FDG AV45)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to use')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of Monte Carlo samples')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Validate
    if args.target in args.available:
        print(f"Error: Target modality {args.target} cannot be in available modalities!")
        exit(1)

    # Run inference
    generated, ground_truth, subject_ids = inference_pipeline(
        target_modality=args.target,
        available_modalities=args.available,
        split=args.split,
        num_samples=args.num_samples,
        device=args.device
    )

    print("\nInference completed!")
    print(f"Generated images: {generated.shape}")
    print(f"Ground truth images: {ground_truth.shape}")
