# 🚀 Project setup prompt for GitHub Copilot
# Goal: Build a minimal, educational Diffusion Model for CFD (Kolmogorov Flow) using PyTorch.
# This project will replicate in simplified form the paper "A Physics-informed Diffusion Model for High-Fidelity Flow Field Reconstruction"
# but without the physics-informed guidance.
# It should be small enough to train on CPU or free Google Colab GPU in one day.

# =============================
# 📁 Repository: diffusionTest
# =============================
# Required folders:
# ├── data/                  # will contain downloaded or synthetic CFD datasets (e.g., Kolmogorov flow)
# ├── models/                # U-Net and diffusion modules
# ├── scripts/               # training, sampling, evaluation scripts
# ├── utils/                 # helper functions: visualization, normalization, etc.
# ├── configs/               # YAML configs for hyperparameters
# ├── checkpoints/           # saved weights
# └── main.py                # entry point for training and sampling

# =============================
# 🧩 Functional requirements
# =============================
# 1. Implement a small U-Net model (PyTorch) for noise prediction εθ(x_t, t, cond)
#    - Input: 6 channels (noisy u,v,p + conditioning Ω, α, Re)
#    - Output: 3 channels (predicted noise)
#    - Depth 3, base_channels=32, no attention for now.
#
# 2. Implement the diffusion process (DDPM core)
#    - Forward diffusion: add noise with linear β schedule
#    - Reverse sampling: denoise using the U-Net
#    - Support T=50 steps for fast training
#
# 3. Training script
#    - Loads data (simple npy/h5 or synthetic if none found)
#    - Randomly samples t, adds noise, trains to predict ε
#    - L2 loss, AdamW optimizer, lr=1e-4
#    - Save checkpoints to /checkpoints
#
# 4. Sampling script
#    - Loads a trained checkpoint and generates flow fields
#    - Plots comparison between noisy, denoised, and ground truth (matplotlib)
#
# 5. Evaluation metrics
#    - MSE between predicted mean and true field
#    - Optional: visualize uncertainty (variance of samples)
#
# 6. Config file (YAML)
#    - Includes: learning_rate, batch_size, T, image_size, channels, base_channels, data_path

# =============================
# 💡 Design goals
# =============================
# - Keep model < 5M parameters
# - Training feasible in < 2 hours on Colab
# - Clean modular code: model, diffusion, train, sample clearly separated
# - Include a simple README.md explaining how to train and sample

# Copilot should now generate the initial project scaffolding with placeholder code files.
# After that, I will refine each file (model, training, diffusion) iteratively.
