import argparse
import torch
import os
import yaml

from diffusion.ddpm import Diffusion
from models.unet import UNet
from utils.io import load_checkpoint
from utils.visualize import compare_fields


def sample(ckpt, out_dir='samples', n_samples=4, config='configs/default.yaml'):
    cfg = yaml.safe_load(open(config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=6, out_channels=3, base_channels=cfg['base_channels'], depth=cfg['depth']).to(device)
    model = load_checkpoint(model, ckpt, map_location=device)
    model.eval()
    diffusion = Diffusion(model, image_size=cfg['image_size'], T=cfg['T'], device=device)

    os.makedirs(out_dir, exist_ok=True)
    samples = diffusion.p_sample_loop((n_samples, 3, cfg['image_size'], cfg['image_size']))
    for i in range(n_samples):
        s = samples[i]
        # no ground truth here; save numpy
        path = os.path.join(out_dir, f'sample_{i}.npy')
        torch.save(s.cpu(), path)
        print('Saved', path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', default='samples')
    args = p.parse_args()
    sample(args.ckpt, args.out)
