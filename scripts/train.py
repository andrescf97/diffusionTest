import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.unet import UNet
from diffusion.ddpm import Diffusion
from utils.io import load_numpy_data, save_checkpoint


def train(config_path):
    cfg = yaml.safe_load(open(config_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load or generate data
    data_path = cfg.get('data_path')
    cond_path = cfg.get('cond_path')
    if not os.path.exists(data_path):
        from scripts.generate_synthetic import make_synthetic
        make_synthetic(data_path, cond_path)

    data, cond = load_numpy_data(data_path, cond_path)
    ds = TensorDataset(data, cond) if cond is not None else TensorDataset(data)
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True)

    model = UNet(in_channels=6, out_channels=3, base_channels=cfg['base_channels'], depth=cfg['depth']).to(device)
    diffusion = Diffusion(model, image_size=cfg['image_size'], T=cfg['T'], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    mse = torch.nn.MSELoss()

    for epoch in range(cfg.get('epochs', 1)):
        for step, batch in enumerate(loader):
            if cond is not None:
                x, c = batch
                # combine cond scalars into channels for simplicity
                b = x.shape[0]
                cond_map = c.view(b, -1, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
                inp = torch.cat([x, cond_map], dim=1).to(device)
            else:
                inp = batch[0].to(device)

            # sample t and noise
            b = inp.shape[0]
            t = torch.randint(0, cfg['T'], (b,), device=device).long()
            x0 = inp[:, :3, :, :]
            xt, noise = diffusion.q_sample(x0, t)

            pred = model(xt)
            loss = mse(pred, noise.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 10 == 0:
                print(f'Epoch {epoch} step {step} loss {loss.item():.4f}')

        # checkpoint
        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = os.path.join('checkpoints', f'epoch{epoch}.pt')
        save_checkpoint(model, opt, ckpt_path)
        print('Saved', ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    train(args.config)
