# ACADiff: Adaptive Clinical-Aware Latent Diffusion

PyTorch implementation of **"Adaptive Clinical-Aware Latent Diffusion for Multimodal Brain Image Generation and Missing Modality Imputation"**.

> **Note:** This is a research prototype. The code is under active development.

---

## Overview

ACADiff synthesizes missing brain imaging modalities using latent diffusion models with adaptive multi-source fusion and clinical-aware conditioning.

**Key Features:**
- Adaptive fusion for 2→1 and 1→1 generation
- Clinical conditioning (MMSE, ADAS13, CDR-SOB)
- DDIM sampling for fast inference
- Supports sMRI, FDG-PET, AV45-PET

---

## Quick Start

### Installation

```bash
conda create -n acadiff python=3.8
conda activate acadiff
pip install torch nibabel pandas numpy scikit-image scikit-learn tqdm
```

### Data Preparation

Place your data in the following structure:
```
data/
├── VBM/
├── FDG/
└── AV45/
sta/
└── cognitive_scores_matched.csv
```

### Training

```bash
# Step 1: Train VAE
python models/vae.py --modality VBM

# Step 2: Train Diffusion
python train_diffusion.py --target VBM

# Step 3: Inference
python inference.py --target VBM --available FDG AV45
```

### Configuration

Edit `config.py` to modify hyperparameters.

---

## Project Structure

```
├── config.py              # Configuration
├── train_diffusion.py     # Training script
├── inference.py           # Inference script
├── models/
│   ├── vae.py            # VAE model + training
│   └── diffusion.py      # Diffusion model
└── utils/
    ├── dataload.py       # Dataset loader
    └── evaluate.py       # Evaluation metrics
```

---

## Citation

```bibtex
@inproceedings{zhou2026acadiff,
  title={Adaptive Clinical-Aware Latent Diffusion for Multimodal Brain Image Generation},
  author={Zhou, Rong and Zhou, Houliang and Su, Yao and Chen, Brian Y. and Zhang, Yu and He, Lifang},
  booktitle={ISBI},
  year={2026}
}
```

---

## License

MIT License

---

## Contact

Rong Zhou - rongzhou7@gmail.com
