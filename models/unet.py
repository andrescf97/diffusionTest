import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Small U-Net. Accepts optional scalar conditioning (packed into channels by caller).

    Args:
        in_channels: number of input channels
        out_channels: output channels
        base_channels: number of base channels
        depth: number of down/up steps
    """

    def __init__(self, in_channels=6, out_channels=3, base_channels=32, depth=3):
        super().__init__()
        self.depth = depth
        enc_channels = [base_channels * (2 ** i) for i in range(depth)]

        # encoder
        self.encs = nn.ModuleList()
        prev = in_channels
        for c in enc_channels:
            self.encs.append(ConvBlock(prev, c))
            prev = c

        # bottleneck
        self.bottleneck = ConvBlock(prev, prev * 2)

        # decoder
        dec_channels = list(reversed(enc_channels))
        self.decs = nn.ModuleList()
        # prev currently equals the last encoder channel; bottleneck expands it
        prev = prev * 2  # bottleneck output channels
        for c in dec_channels:
            in_ch = prev + c  # because we'll concat skip connections (prev from up, c from skip)
            self.decs.append(ConvBlock(in_ch, c))
            prev = c

        self.final = nn.Conv2d(prev, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t=None, cond=None):
        # x: [B, C, H, W]
        # If t or cond are provided as scalars per sample, caller should have concatenated them into x.
        skips = []
        h = x
        for enc in self.encs:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decs, reversed(skips)):
            h = self.up(h)
            # crop/pad if necessary (shapes should match with power-of-two sizes)
            if h.shape[-2:] != skip.shape[-2:]:
                h = nn.functional.interpolate(h, size=skip.shape[-2:], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        return self.final(h)


if __name__ == '__main__':
    # simple smoke test
    m = UNet(in_channels=6, out_channels=3, base_channels=16, depth=3)
    x = torch.randn(2, 6, 64, 64)
    y = m(x)
    print('out', y.shape)
