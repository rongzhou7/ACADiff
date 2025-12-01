"""
Configuration file for ACADiff
All hyperparameters and paths in one place
"""
import os


class Config:
    """
    Global configuration for the entire project
    """

    # ========== Paths ==========
    DATA_ROOT = './data'
    CSV_PATH = './sta/cognitive_scores_matched.csv'
    SAVE_DIR = './checkpoints'
    LOG_DIR = './logs'

    # ========== Data ==========
    MODALITIES = ['VBM', 'FDG', 'AV45']  # Folder names
    TARGET_SHAPE = (160, 180, 160)  # Input image shape
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    # TEST_RATIO = 0.1 (implicit)

    # ========== VAE ==========
    # Architecture
    VAE_LATENT_CHANNELS = 8  # Paper uses 8 channels
    VAE_LATENT_SHAPE = (20, 22, 20)  # Compressed spatial dimensions
    VAE_BASE_CHANNELS = 64  # Base number of channels in encoder/decoder
    VAE_CH_MULT = [1, 2, 4, 8]  # Channel multipliers for each layer
    VAE_NUM_RES_BLOCKS = 2  # Number of residual blocks per resolution

    # Training
    VAE_BATCH_SIZE = 2  # Adjust based on GPU memory
    VAE_LEARNING_RATE = 1e-4
    VAE_NUM_EPOCHS = 100
    VAE_KL_WEIGHT = 1e-6  # Weight for KL divergence loss
    VAE_GRAD_CLIP = 1.0  # Gradient clipping

    # ========== Diffusion ==========
    # U-Net architecture
    UNET_BASE_CHANNELS = 128
    UNET_CH_MULT = [1, 2, 4, 8]
    UNET_NUM_RES_BLOCKS = 2
    UNET_ATTN_RESOLUTIONS = [16]  # Which resolutions to use attention

    # Diffusion process
    DIFFUSION_TIMESTEPS = 1000  # T in paper
    DIFFUSION_BETA_SCHEDULE = 'cosine'  # 'linear' or 'cosine'
    DIFFUSION_BETA_START = 0.0001
    DIFFUSION_BETA_END = 0.02

    # Training
    DIFF_BATCH_SIZE = 2
    DIFF_LEARNING_RATE = 1e-4
    DIFF_NUM_EPOCHS = 200
    DIFF_CONSISTENCY_WEIGHT = 0.1  # Lambda for L_cons
    DIFF_GRAD_CLIP = 1.0

    # Modality dropout (for adaptive training)
    MODALITY_DROPOUT_PROB = 0.5  # Probability of using 1->1 instead of 2->1
    CLINICAL_DROPOUT_PROB = 0.1  # Probability of dropping clinical info

    # ========== Clinical Encoding ==========
    USE_GPT4O = True  # True: use GPT-4o, False: use learnable embeddings
    GPT4O_MODEL = 'gpt-4o'  # If using GPT-4o
    CLINICAL_EMBED_DIM = 512  # Embedding dimension for clinical data

    # ========== Inference ==========
    INFERENCE_NUM_STEPS = 50  # DDIM sampling steps (paper uses fewer than training)
    INFERENCE_BATCH_SIZE = 1
    INFERENCE_NUM_SAMPLES = 10  # Monte Carlo sampling (paper uses 10)

    # ========== Evaluation ==========
    EVAL_BATCH_SIZE = 4
    EVAL_METRICS = ['MAE', 'PSNR', 'SSIM', 'NMI']

    # Classification (downstream task)
    CLASSIFIER_MODEL = 'densenet121'
    CLASSIFIER_BATCH_SIZE = 8
    CLASSIFIER_LEARNING_RATE = 1e-4
    CLASSIFIER_NUM_EPOCHS = 50

    # ========== Hardware ==========
    NUM_WORKERS = 4  # DataLoader workers
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    GPU_IDS = [0, 1, 2, 3]  # Which GPUs to use if multi-GPU
    USE_AMP = True  # Automatic Mixed Precision

    # ========== Logging ==========
    LOG_INTERVAL = 10  # Print every N batches
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    EVAL_INTERVAL = 5  # Evaluate every N epochs

    # ========== Reproducibility ==========
    SEED = 42

    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.SAVE_DIR, 'vae'), exist_ok=True)
        os.makedirs(os.path.join(cls.SAVE_DIR, 'diffusion'), exist_ok=True)
        os.makedirs(os.path.join(cls.SAVE_DIR, 'classifier'), exist_ok=True)

    @classmethod
    def print_config(cls):
        """Print all configuration"""
        print("="*60)
        print("Configuration")
        print("="*60)
        for key in dir(cls):
            if not key.startswith('_') and key.isupper():
                print(f"{key:30s}: {getattr(cls, key)}")
        print("="*60)


# Alternative: use a dict if you prefer
CONFIG_DICT = {
    'data': {
        'data_root': './data',
        'csv_path': './sta/cognitive_scores_matched.csv',
        'modalities': ['VBM', 'FDG', 'AV45'],
        'target_shape': (160, 180, 160),
        'train_ratio': 0.8,
        'val_ratio': 0.1,
    },
    'vae': {
        'latent_channels': 8,
        'latent_shape': (20, 22, 20),
        'base_channels': 64,
        'ch_mult': [1, 2, 4, 8],
        'batch_size': 2,
        'lr': 1e-4,
        'num_epochs': 100,
    },
    'diffusion': {
        'timesteps': 1000,
        'beta_schedule': 'cosine',
        'batch_size': 2,
        'lr': 1e-4,
        'num_epochs': 200,
    },
}


if __name__ == '__main__':
    # Test configuration
    Config.print_config()
    Config.create_dirs()
    print("\nDirectories created successfully!")
