import math
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, temb_dim=0):
        super().__init__()
        # use circular padding for periodic BCs
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='circular')
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, padding_mode='circular')
        self.act2 = nn.ReLU(inplace=True)

        # FiLM (scale & shift) from time embedding -> 2*out_c parameters
        self.temb_dim = temb_dim
        if temb_dim and temb_dim > 0:
            self.film = nn.Linear(temb_dim, out_c * 2)
        else:
            self.film = None

    def forward(self, x, temb=None):
        h = self.conv1(x)
        h = self.act1(h)

        h = self.conv2(h)
        # apply FiLM if available
        if self.film is not None and temb is not None:
            # temb: [B, temb_dim]
            film_params = self.film(temb)  # [B, 2*out_c]
            B = film_params.shape[0]
            C = h.shape[1]
            gamma, beta = film_params.chunk(2, dim=1)
            gamma = gamma.view(B, C, 1, 1)
            beta = beta.view(B, C, 1, 1)
            h = gamma * h + beta

        h = self.act2(h)
        return h


class UNet(nn.Module):
    """Small U-Net with sinusoidal time embedding + FiLM conditioning.

    Args:
        in_channels: number of input channels
        out_channels: output channels
        base_channels: number of base channels
        depth: number of down/up steps
        temb_dim: dimension of time embedding (internal)
    """

    def __init__(self, in_channels=6, out_channels=3, base_channels=32, depth=3, temb_dim=None):
        super().__init__()
        self.depth = depth
        # choose a reasonable default for time-embedding dim
        if temb_dim is None:
            temb_dim = base_channels * 4
        self.temb_dim = temb_dim

        enc_channels = [base_channels * (2 ** i) for i in range(depth)]

        # time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(self.temb_dim, self.temb_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.temb_dim * 4, self.temb_dim),
        )

        # encoder
        self.encs = nn.ModuleList()
        prev = in_channels
        for c in enc_channels:
            self.encs.append(ConvBlock(prev, c, temb_dim=self.temb_dim))
            prev = c

        # bottleneck
        self.bottleneck = ConvBlock(prev, prev * 2, temb_dim=self.temb_dim)

        # decoder
        dec_channels = list(reversed(enc_channels))
        self.decs = nn.ModuleList()
        # prev currently equals the last encoder channel; bottleneck expands it
        prev = prev * 2  # bottleneck output channels
        for c in dec_channels:
            in_ch = prev + c  # because we'll concat skip connections (prev from up, c from skip)
            self.decs.append(ConvBlock(in_ch, c, temb_dim=self.temb_dim))
            prev = c

        self.final = nn.Conv2d(prev, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def sinusoidal_embedding(self, t, dim):
        # t: [B]
        half = dim // 2
        device = t.device
        freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -(math.log(10000) / half))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=device)], dim=1)
        return emb

    def forward(self, x, t=None, cond=None):
        # x: [B, C, H, W]
        # t: [B] integer timesteps (0..T-1)
        # cond: optional channel tensor [B, Cc, H, W]
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        # compute time embedding if provided
        temb = None
        if t is not None:
            temb = self.sinusoidal_embedding(t, self.temb_dim)
            temb = self.time_mlp(temb)

        skips = []
        h = x
        for enc in self.encs:
            h = enc(h, temb)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h, temb)

        for dec, skip in zip(self.decs, reversed(skips)):
            h = self.up(h)
            # crop/pad if necessary (shapes should match with power-of-two sizes)
            if h.shape[-2:] != skip.shape[-2:]:
                h = nn.functional.interpolate(h, size=skip.shape[-2:], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            h = dec(h, temb)

        return self.final(h)


if __name__ == '__main__':
    # simple smoke test
    m = UNet(in_channels=6, out_channels=3, base_channels=16, depth=3)
    x = torch.randn(2, 6, 64, 64)
    y = m(x)
    print('out', y.shape)
