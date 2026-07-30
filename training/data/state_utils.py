"""StatePlay state utilities for state-aware training.

This module keeps StatePlay state loading separate from action preprocessing.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch


_ALIASES = {
    "hp1": ("hp1", "p1_hp"),
    "hp2": ("hp2", "p2_hp"),
    "meter1": ("meter1", "p1_meter"),
    "meter2": ("meter2", "p2_meter"),
    "timer": ("timer",),
}


def parse_state_columns(columns: str) -> list[str]:
    cols = [c.strip() for c in columns.split(",") if c.strip()]
    if len(cols) != 5:
        raise ValueError(
            f"--state_columns must contain exactly 5 comma-separated columns; got {cols!r}"
        )
    return cols


def parse_state_norm_max(norm_max: str) -> torch.Tensor:
    vals = [float(x.strip()) for x in norm_max.split(",") if x.strip()]
    if len(vals) != 5:
        raise ValueError(
            f"--state_norm_max must contain exactly 5 comma-separated values; got {vals!r}"
        )
    if any(v <= 0 for v in vals):
        raise ValueError(f"--state_norm_max values must be positive; got {vals!r}")
    return torch.tensor(vals, dtype=torch.float32)


def _resolve_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    actual = set(df.columns)
    resolved = []
    missing = []
    for col in requested:
        aliases = _ALIASES.get(col, (col,))
        hit = next((name for name in aliases if name in actual), None)
        if hit is None:
            missing.append((col, aliases))
        else:
            resolved.append(hit)
    if missing:
        details = "; ".join(
            f"{col} (accepted: {', '.join(aliases)})" for col, aliases in missing
        )
        raise KeyError(
            "[StatePlay state] Required state columns are missing from action/state parquet: "
            f"{details}. Available columns: {sorted(df.columns.tolist())}"
        )
    return resolved


def load_state_from_action_parquet(
    rel_path: str,
    base_path: str,
    num_frames: int,
    state_columns: list[str],
    state_norm_max: torch.Tensor,
) -> torch.Tensor:
    """Load normalized state as [1, T, 5] float32 from an action parquet."""
    full = os.path.join(base_path, rel_path)
    df = pd.read_parquet(full)
    cols = _resolve_columns(df, state_columns)
    arr = df[cols].values[:num_frames].astype(np.float32)
    if arr.shape[0] == 0:
        raise ValueError(f"[StatePlay state] {full} has no state rows.")
    state = torch.from_numpy(arr)
    norm = state_norm_max.to(dtype=state.dtype)
    state = torch.minimum(torch.clamp(state, min=0.0), norm) / norm
    state = state * 2.0 - 1.0
    return state.unsqueeze(0)


class StateAugmentedDataset(torch.utils.data.Dataset):
    """Wrap a StatePlay dataset and append state loaded from the action path."""

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        metadata_path: str,
        base_path: str,
        num_frames: int,
        state_columns: str,
        state_norm_max: str,
    ):
        self.base_dataset = base_dataset
        self.base_path = base_path
        self.num_frames = num_frames
        self.state_columns = parse_state_columns(state_columns)
        self.state_norm_max = parse_state_norm_max(state_norm_max)
        df = pd.read_csv(metadata_path)
        if "action" not in df.columns:
            raise KeyError(
                "[StatePlay state] metadata must contain an 'action' column "
                "so state can be loaded from the same parquet as actions."
            )
        self.action_paths = df["action"].tolist()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        data = self.base_dataset[idx]
        rel_path = self.action_paths[idx % len(self.action_paths)]
        data["state"] = load_state_from_action_parquet(
            rel_path=rel_path,
            base_path=self.base_path,
            num_frames=self.num_frames,
            state_columns=self.state_columns,
            state_norm_max=self.state_norm_max,
        )
        return data

    @property
    def load_from_cache(self) -> bool:
        return getattr(self.base_dataset, "load_from_cache", False)
