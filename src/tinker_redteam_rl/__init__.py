from __future__ import annotations

from typing import TYPE_CHECKING

from .config import Config

__all__ = ["Config", "RedteamTrainer", "main"]

if TYPE_CHECKING:
    from .rl_loop import main
    from .trainer import RedteamTrainer


def __getattr__(name: str):
    if name == "RedteamTrainer":
        from .trainer import RedteamTrainer

        return RedteamTrainer
    if name == "main":
        from .rl_loop import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
