import os, argparse, numpy as np
from skimage.transform import resize

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x

try:
    import h5py
except Exception:
    h5py = None

def to_hwc(arr: np.ndarray) -> np.ndarray:
    """Ensure array is (H, W, C). Accepts (H,W), (H,W,C), or (C,H,W).
    More robust detection than previous heuristic.
    """
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3:
        # Heuristics to detect channel ordering for 3D arrays.
        # If the first axis is larger than the other two, it's likely channels-first (C,H,W).
        if arr.shape[0] > arr.shape[1] and arr.shape[0] > arr.shape[2]:
            arr = np.moveaxis(arr, 0, -1)
        # If the last axis is small (<=8), it's likely channels-last (H,W,C).
        elif arr.shape[-1] <= 8:
            pass
        # If the first axis is small, also treat as channels-first.
        elif arr.shape[0] <= 8:
            arr = np.moveaxis(arr, 0, -1)
        else:
            # fallback: assume H,W,C
            pass
    else:
        raise ValueError(f"Unexpected shape: {arr.shape}")
    return arr

def downsample_field(field: np.ndarray, out_size: int) -> np.ndarray:
    """Resize a single field to (out_size, out_size, C)."""
    field = to_hwc(field)
    # Use a simple call compatible with older and newer scikit-image: resize
    # will operate on the full array shaped (H,W,C). Newer versions accept
    # `channel_axis`, but older ones expect the channels to be included in
    # the shape and don't need an extra kwarg. This call works across versions.
    out = resize(
        field,
        (out_size, out_size, field.shape[-1]),
        anti_aliasing=True,
        preserve_range=True,
    )
    return out.astype(np.float32)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    path = args.hi_res_path
    data = None
    loader_type = None

    if path.endswith((".h5", ".hdf5")):
        if h5py is None:
            raise RuntimeError("h5py not installed; please `pip install h5py` to load .h5 files")
        f = h5py.File(path, "r")
        key = args.key or list(f.keys())[0]
        data = f[key]
        loader_type = "h5"
    elif path.endswith(".npz"):
        npz = np.load(path, allow_pickle=True)
        key = args.key or (npz.files[0])
        data = npz[key]
        loader_type = "npz"
    elif path.endswith(".npy"):
        data = np.load(path, mmap_mode="r")
        loader_type = "npy"
    else:
        raise ValueError("Unsupported file type: must be .npy, .npz or .h5")

    n_total = int(data.shape[0])
    N = min(args.max_samples, n_total)
    print(f"Loaded {n_total} samples (loader={loader_type}), processing {N}. Example shape: {tuple(data[0].shape)}")

    # process in batches to reduce peak memory
    batch = args.batch_size
    out_list = []
    for i0 in tqdm(range(0, N, batch), desc="downsampling"):
        i1 = min(N, i0 + batch)
        chunk = []
        for i in range(i0, i1):
            arr = data[i]
            # ensure numpy array (h5 datasets may return scalar wrappers)
            arr = np.array(arr)
            chunk.append(downsample_field(arr, args.size))
        out_list.append(np.stack(chunk, axis=0))

    downs = np.concatenate(out_list, axis=0)

    # Save in HWC or CHW layout
    if args.channels_first:
        downs = np.moveaxis(downs, -1, 1)  # (N,C,H,W)
    out_name = f"downsampled_{args.size}{'_chw' if args.channels_first else ''}.npz"
    out_path = os.path.join(args.out_dir, out_name)
    np.savez_compressed(out_path, **{args.out_key: downs})
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hi_res_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="data/processed")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--key", type=str, default=None)
    p.add_argument("--channels_first", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_key", type=str, default="x")
    args = p.parse_args()
    main(args)
