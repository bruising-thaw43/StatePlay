"""Action loading: parquet → keyboard tensor, with hold-window densification."""
import numpy as np
import pandas as pd
import torch

from ..constants import SF3_BUTTON_COLS


def hold_last_upsample(btn: torch.Tensor, window: int) -> torch.Tensor:
    """Forward-broadcast each `window`-sized bucket's start value over its siblings.

    Mirrors training-time densification: parquet sampled at 10Hz, video at 20fps,
    so half the rows would be zeros — this fills them by holding the bucket's
    first value (including legitimate zeros).
    """
    if window <= 1:
        return btn
    T = btn.shape[0]
    idx = (torch.arange(T) // window) * window
    idx = idx.clamp(max=T - 1)
    return btn[idx]


def load_actions(parquet_path: str, num_frames: int, hold_window: int = 10,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 button_cols=None) -> torch.Tensor:
    """Read a parquet, slice to `num_frames`, hold-upsample, return [1, T, K]."""
    df = pd.read_parquet(parquet_path)
    cols = list(button_cols or SF3_BUTTON_COLS)
    arrs = []
    for col in cols:
        if col in df.columns:
            arrs.append(df[col].values[:num_frames].astype(np.float32))
        else:
            arrs.append(np.zeros((min(num_frames, len(df)),), dtype=np.float32))
    arr = np.stack(arrs, axis=1)
    btn = torch.tensor(arr)
    btn = hold_last_upsample(btn, hold_window)
    return btn.unsqueeze(0).to(device, dtype)
