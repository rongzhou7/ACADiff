"""
ADNI Multimodal Dataset
Load sMRI, FDG-PET, AV45-PET three modalities
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


class ADNIDataset(Dataset):
    """
    ADNI three-modality dataset
    """
    def __init__(self,
                 data_root=None,
                 csv_path=None,
                 modalities=None,
                 target_shape=None,
                 split='train',
                 train_ratio=None,
                 val_ratio=None):
        """
        Args:
            data_root: Data root directory containing VBM/, FDG/, AV45/ folders
            csv_path: Path to cognitive_scores_matched.csv
            modalities: List of modality names
            target_shape: Target image shape
            split: 'train', 'val', 'test'
        """
        # Use Config defaults if not specified
        data_root = data_root or Config.DATA_ROOT
        csv_path = csv_path or Config.CSV_PATH
        modalities = modalities or Config.MODALITIES
        target_shape = target_shape or Config.TARGET_SHAPE
        train_ratio = train_ratio or Config.TRAIN_RATIO
        val_ratio = val_ratio or Config.VAL_RATIO
        self.data_root = data_root
        self.modalities = modalities
        self.target_shape = target_shape

        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Get unique subject IDs
        unique_subjects = self.df['Subject_ID'].unique()
        n_subjects = len(unique_subjects)

        # Data split
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)

        if split == 'train':
            subjects = unique_subjects[:n_train]
        elif split == 'val':
            subjects = unique_subjects[n_train:n_train+n_val]
        else:  # test
            subjects = unique_subjects[n_train+n_val:]

        # Filter data for current split
        # For each subject, only take the first visit with complete scores
        self.samples = []
        for subject_id in subjects:
            subject_data = self.df[self.df['Subject_ID'] == subject_id]
            # Find first record with all three scores
            complete_data = subject_data.dropna(subset=['cdr_sb', 'adas13', 'MMSE'])
            if len(complete_data) > 0:
                first_visit = complete_data.iloc[0]
                self.samples.append({
                    'subject_id': subject_id,
                    'session_id': first_visit['session_id'],
                    'cdr_sb': first_visit['cdr_sb'],
                    'adas13': first_visit['adas13'],
                    'mmse': first_visit['MMSE']
                })

        print(f"{split} set: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _load_nifti(self, modality, subject_id, session_id):
        """
        Load NIfTI file
        Assumed naming: {modality}/{subject_id}_{session_id}.nii.gz
        Modify pattern here if your naming is different
        """
        # Try several common naming patterns
        patterns = [
            f"{subject_id}_{session_id}.nii.gz",
            f"{subject_id}_{session_id}.nii",
            f"sub-{subject_id}_ses-{session_id}.nii.gz",
        ]

        modality_dir = os.path.join(self.data_root, modality)

        for pattern in patterns:
            filepath = os.path.join(modality_dir, pattern)
            if os.path.exists(filepath):
                img = nib.load(filepath)
                data = img.get_fdata()
                return data

        # If not found, list all files and try to match subject_id
        if os.path.exists(modality_dir):
            files = os.listdir(modality_dir)
            for f in files:
                if subject_id.replace('_', '') in f.replace('_', ''):
                    filepath = os.path.join(modality_dir, f)
                    img = nib.load(filepath)
                    data = img.get_fdata()
                    return data

        raise FileNotFoundError(f"Cannot find file for {modality}/{subject_id}_{session_id}")

    def _normalize(self, data):
        """
        Normalize to [-1, 1]
        """
        # First normalize to [0, 1]
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        # Then map to [-1, 1]
        data = data * 2.0 - 1.0
        return data

    def _resize_if_needed(self, data):
        """
        Simple crop or pad if size doesn't match
        Assumes data is already correct size; add resize logic if needed
        """
        current_shape = data.shape
        if current_shape != self.target_shape:
            # Simple center crop/pad
            # May need more complex handling in practice
            print(f"Warning: shape mismatch {current_shape} vs {self.target_shape}")
        return data

    def generate_prompt(self, sample):
        """
        Generate clinical prompt text
        Format: "Generate [TARGET] from [AVAILABLE] for patient with MMSE=X, ADAS13=Y, CDR-SOB=Z"
        Generate generic description here; TARGET/AVAILABLE filled dynamically during training
        """
        prompt = (f"Patient with MMSE={sample['mmse']:.0f}, "
                  f"ADAS13={sample['adas13']:.1f}, "
                  f"CDR-SOB={sample['cdr_sb']:.1f}")
        return prompt

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load three modalities
        images = {}
        for modality in self.modalities:
            try:
                data = self._load_nifti(modality, sample['subject_id'], sample['session_id'])
                data = self._resize_if_needed(data)
                data = self._normalize(data)
                # Convert to torch tensor: [1, D, H, W] (add channel dimension)
                images[modality] = torch.FloatTensor(data).unsqueeze(0)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                # Fill with zeros if not found
                images[modality] = torch.zeros(1, *self.target_shape)

        # Generate prompt
        prompt = self.generate_prompt(sample)

        # Cognitive scores
        scores = {
            'mmse': sample['mmse'],
            'adas13': sample['adas13'],
            'cdr_sb': sample['cdr_sb']
        }

        return {
            'images': images,  # dict: {'VBM': tensor, 'FDG': tensor, 'AV45': tensor}
            'prompt': prompt,
            'scores': scores,
            'subject_id': sample['subject_id'],
            'session_id': sample['session_id']
        }


def get_dataloaders(batch_size=None, num_workers=None):
    """
    Create train, val, test DataLoaders
    """
    batch_size = batch_size or Config.VAE_BATCH_SIZE
    num_workers = num_workers or Config.NUM_WORKERS

    train_dataset = ADNIDataset(split='train')
    val_dataset = ADNIDataset(split='val')
    test_dataset = ADNIDataset(split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# Test code
if __name__ == '__main__':
    print("Testing ADNI Dataset...")

    # Create dataset
    dataset = ADNIDataset(split='train')
    print(f"Total samples: {len(dataset)}")

    # Load one sample
    sample = dataset[0]
    print(f"\nSample structure:")
    print(f"  Subject ID: {sample['subject_id']}")
    print(f"  Session ID: {sample['session_id']}")
    print(f"  Prompt: {sample['prompt']}")
    print(f"  Scores: {sample['scores']}")
    print(f"  Images:")
    for modality, img in sample['images'].items():
        print(f"    {modality}: {img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")

    # Test DataLoader
    print("\nTesting DataLoader...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=2, num_workers=0)

    batch = next(iter(train_loader))
    print(f"Batch size: {len(batch['subject_id'])}")
    print(f"VBM batch shape: {batch['images']['VBM'].shape}")
