import os
import numpy as np
import torch


def load_numpy_data(data_path, cond_path=None):
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    data = np.load(data_path)
    data = torch.from_numpy(data).float()
    cond = None
    if cond_path and os.path.exists(cond_path):
        cond = torch.from_numpy(np.load(cond_path)).float()
    return data, cond


def save_checkpoint(model, opt, path):
    sd = {
        'model_state': model.state_dict(),
        'opt_state': opt.state_dict() if opt is not None else None,
    }
    torch.save(sd, path)


def load_checkpoint(model, path, map_location='cpu'):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck['model_state'])
    return model
