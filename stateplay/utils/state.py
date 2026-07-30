"""State loading and denormalization for StatePlay inference."""
from pathlib import Path
from typing import Sequence

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

STATE_COLUMNS = ("timer", "hp1", "hp2", "meter1", "meter2")
STATE_NORM_MAX = torch.tensor([99.0, 160.0, 160.0, 104.0, 96.0], dtype=torch.float32)
_ALIASES = {
    "timer": ("timer",),
    "hp1": ("hp1", "p1_hp"),
    "hp2": ("hp2", "p2_hp"),
    "meter1": ("meter1", "p1_meter"),
    "meter2": ("meter2", "p2_meter"),
}


def _resolve_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    available = set(df.columns)
    resolved = []
    missing = []
    for col in columns:
        aliases = _ALIASES.get(col, (col,))
        hit = next((name for name in aliases if name in available), None)
        if hit is None:
            missing.append((col, aliases))
        else:
            resolved.append(hit)
    if missing:
        details = "; ".join(
            f"{col} (accepted: {', '.join(aliases)})" for col, aliases in missing
        )
        raise KeyError(
            "Required StatePlay state columns are missing: "
            f"{details}. Available columns: {sorted(df.columns.tolist())}"
        )
    return resolved


def normalize_state(raw_state: torch.Tensor, norm_max: torch.Tensor = STATE_NORM_MAX) -> torch.Tensor:
    norm = norm_max.to(device=raw_state.device, dtype=raw_state.dtype)
    state = torch.minimum(torch.clamp(raw_state, min=0.0), norm) / norm
    return state * 2.0 - 1.0


def denormalize_state(norm_state: torch.Tensor, norm_max: torch.Tensor = STATE_NORM_MAX) -> torch.Tensor:
    norm = norm_max.to(device=norm_state.device, dtype=norm_state.dtype)
    return (norm_state + 1.0) * 0.5 * norm


def load_initial_state(
    parquet_path: str,
    num_frames: int,
    latent_frames: int,
    sampling: str = "interpolation",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    columns: Sequence[str] = STATE_COLUMNS,
) -> torch.Tensor:
    """Return the first normalized latent-state token as [1, 1, 5]."""
    df = pd.read_parquet(parquet_path)
    cols = _resolve_columns(df, columns)
    arr = df[cols].values[:num_frames].astype(np.float32)
    if arr.shape[0] == 0:
        raise ValueError(f"{parquet_path} contains no state rows.")
    state = normalize_state(torch.from_numpy(arr))
    if sampling == "interpolation":
        initial = state[:1]
    elif sampling == "end":
        end_idx = math.ceil(state.shape[0] / latent_frames) - 1
        initial = state[end_idx:end_idx + 1]
    elif sampling == "avg":
        end_idx = math.ceil(state.shape[0] / latent_frames)
        initial = state[:end_idx].mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"unsupported state sampling: {sampling!r}")
    return initial.unsqueeze(0).to(device=device, dtype=dtype)


def downsample_state_to_latent(
    state: torch.Tensor,
    latent_frames: int,
    sampling: str = "interpolation",
) -> torch.Tensor:
    """Downsample normalized frame-level state [B, T, 5] to latent frames [B, F, 5]."""
    if state.ndim == 4 and state.shape[1] == 1:
        state = state[:, 0]
    if state.ndim != 3 or state.shape[-1] != len(STATE_COLUMNS):
        raise ValueError(
            f"expected state shape [B, T, {len(STATE_COLUMNS)}] or [B, 1, T, {len(STATE_COLUMNS)}], "
            f"got {tuple(state.shape)}"
        )
    if state.shape[1] == latent_frames:
        return state
    if sampling == "interpolation":
        return F.interpolate(
            state.transpose(1, 2).float(), size=latent_frames, mode="linear", align_corners=True,
        ).transpose(1, 2).to(dtype=state.dtype)
    if sampling == "avg":
        return F.adaptive_avg_pool1d(
            state.transpose(1, 2).float(), output_size=latent_frames,
        ).transpose(1, 2).to(dtype=state.dtype)
    if sampling == "end":
        total_frames = state.shape[1]
        end_idx = torch.ceil(
            torch.arange(1, latent_frames + 1, device=state.device, dtype=torch.float32)
            * total_frames
            / latent_frames
        ).long() - 1
        end_idx = end_idx.clamp(min=0, max=total_frames - 1)
        return state.index_select(1, end_idx)
    raise ValueError(f"unsupported state sampling: {sampling!r}")


def load_true_state(
    parquet_path: str,
    num_frames: int,
    latent_frames: int,
    sampling: str = "interpolation",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    columns: Sequence[str] = STATE_COLUMNS,
) -> torch.Tensor:
    """Return normalized latent-frame ground-truth state as [1, F, 5]."""
    df = pd.read_parquet(parquet_path)
    cols = _resolve_columns(df, columns)
    arr = df[cols].values[:num_frames].astype(np.float32)
    if arr.shape[0] == 0:
        raise ValueError(f"{parquet_path} contains no state rows.")
    state = normalize_state(torch.from_numpy(arr)).unsqueeze(0)
    state = downsample_state_to_latent(state, latent_frames=latent_frames, sampling=sampling)
    return state.to(device=device, dtype=dtype)


def save_state_txt(
    state: torch.Tensor,
    path: str,
    columns: Sequence[str] = STATE_COLUMNS,
    true_state: torch.Tensor | None = None,
) -> None:
    """Save normalized state after inverse scaling, optionally with true/error columns."""
    pred = denormalize_state(state.detach().float().cpu()).squeeze(0)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if true_state is not None:
        true = denormalize_state(true_state.detach().float().cpu()).squeeze(0)
        if true.shape != pred.shape:
            raise ValueError(f"true_state shape {tuple(true.shape)} does not match pred shape {tuple(pred.shape)}")
        err = (pred - true).abs()
        with out.open("w", encoding="utf-8") as f:
            f.write(f"{'frame':>5}")
            for col in columns:
                f.write(f" {col + '_pred':>12} {col + '_true':>12} {col + '_err':>12}")
            f.write("\n")
            for i in range(pred.shape[0]):
                f.write(f"{i:5d}")
                for j in range(len(columns)):
                    f.write(
                        f" {pred[i, j].item():12.3f}"
                        f" {true[i, j].item():12.3f}"
                        f" {err[i, j].item():12.3f}"
                    )
                f.write("\n")
        return

    raw = pred.numpy()
    header = "{:>12} ".format("latent_frame") + " ".join("{:>12}".format(col) for col in columns)
    with out.open("w") as f:
        f.write(header + "\n")
        for i, row in enumerate(raw):
            values = " ".join("{:12.6f}".format(float(x)) for x in row)
            f.write("{:12d} {}\n".format(i, values))

