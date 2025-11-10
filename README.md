# diffusionTest

Minimal educational diffusion model for CFD (Kolmogorov flow) using PyTorch.

This repository contains a lightweight DDPM-style implementation and a small U-Net intended to run on CPU or a free Colab GPU within a day.

Quickstart
- Create a Python environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

- Generate synthetic data (optional):

```bash
python main.py generate --out data/synthetic
```

- Train a small model with the default config:

```bash
python main.py train --config configs/default.yaml
```

- Sample from a checkpoint:

```bash
python main.py sample --ckpt checkpoints/latest.pt --out samples
```

Project layout
- `models/` - U-Net implementation
- `diffusion/` - DDPM forward/reverse logic
- `scripts/` - training, sampling and synthetic data generator
- `utils/` - IO and visualization helpers
- `configs/` - default YAML config
- `data/`, `checkpoints/` - storage for dataset and weights

Notes
- This is an educational, minimal scaffold. Replace the synthetic data loader with real CFD data when available. The default settings keep the model small (depth=3, base_channels=32) and T=50 steps for fast iteration.
