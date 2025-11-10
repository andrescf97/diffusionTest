import matplotlib.pyplot as plt
import numpy as np


def compare_fields(x_noisy, x_denoised, x_gt=None, out_path=None):
    # Plot first sample's channels side by side
    # x_*: torch or numpy arrays [C,H,W]
    if hasattr(x_noisy, 'cpu'):
        x_noisy = x_noisy.cpu().numpy()
    if hasattr(x_denoised, 'cpu'):
        x_denoised = x_denoised.cpu().numpy()
    if x_gt is not None and hasattr(x_gt, 'cpu'):
        x_gt = x_gt.cpu().numpy()

    C = x_noisy.shape[0]
    fig, axs = plt.subplots(C, 3 if x_gt is not None else 2, figsize=(4 * (3 if x_gt is not None else 2), 3 * C))
    for c in range(C):
        axs[c, 0].imshow(x_noisy[c], cmap='RdBu')
        axs[c, 0].set_title(f'noisy ch{c}')
        axs[c, 1].imshow(x_denoised[c], cmap='RdBu')
        axs[c, 1].set_title(f'denoised ch{c}')
        if x_gt is not None:
            axs[c, 2].imshow(x_gt[c], cmap='RdBu')
            axs[c, 2].set_title(f'gt ch{c}')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    return fig
