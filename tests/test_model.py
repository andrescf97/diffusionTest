import torch
from models.unet import UNet


def test_unet_shape():
    m = UNet(in_channels=6, out_channels=3, base_channels=16, depth=3)
    x = torch.randn(1, 6, 64, 64)
    y = m(x)
    assert y.shape == (1, 3, 64, 64)


if __name__ == '__main__':
    test_unet_shape()
    print('smoke test passed')
