import torch
import torch.nn as nn
import numpy as np


class Diffusion:
    def __init__(self, model, image_size=64, T=50, device='cpu'):
        self.model = model
        self.image_size = image_size
        self.T = T
        self.device = device

        # linear beta schedule
        betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        self.betas = torch.from_numpy(betas).to(device)
        self.alphas = torch.from_numpy(alphas).to(device)
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod).to(device)

    def q_sample(self, x0, t, noise=None):
        # x0: [B,C,H,W], t: [B] long
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_1m_acp = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None]
        return sqrt_acp * x0 + sqrt_1m_acp * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None):
        device = self.device
        b = shape[0]
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((b,), i, dtype=torch.long, device=device)
            eps = self.model(x)
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_cum = self.alphas_cumprod[i]
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            coef1 = 1 / torch.sqrt(alpha)
            coef2 = (1 - alpha) / torch.sqrt(1 - alpha_cum)
            x = coef1 * (x - coef2 * eps) + torch.sqrt(beta) * noise
        return x
