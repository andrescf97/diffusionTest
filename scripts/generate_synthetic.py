import numpy as np
import os


def make_synthetic(out_path='data/synthetic.npy', cond_path='data/cond.npy', N=32, C=3, H=64, W=64):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = np.random.randn(N, C, H, W).astype(np.float32)
    cond = np.random.randn(N, 3).astype(np.float32)  # Omega, alpha, Re
    np.save(out_path, data)
    np.save(cond_path, cond)
    print(f'Wrote synthetic data to {out_path} and cond to {cond_path}')


if __name__ == '__main__':
    make_synthetic()
