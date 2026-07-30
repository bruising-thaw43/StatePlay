"""StatePlay game and action configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GameProfile:
    name: str
    description: str
    button_cols: Tuple[str, ...]
    fixed_prompt: str
    default_height: int = 480
    default_width: int = 832
    default_num_frames: int = 101
    default_use_csv_prompt: bool = True
    default_action_hold_window: int = 10

    @property
    def num_buttons(self) -> int:
        return len(self.button_cols)


SF3 = GameProfile(
    name="sf3",
    description="Street Fighter III (CPS3): Ryu vs Ibuki @ Ibuki's stage",
    button_cols=("UP", "DOWN", "LEFT", "RIGHT", "Y", "X", "Z", "A", "B", "C", "D"),
    fixed_prompt="SF3 Game.",
)
