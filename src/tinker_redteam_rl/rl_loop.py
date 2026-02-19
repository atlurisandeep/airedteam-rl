from __future__ import annotations

from .config import Config
from .trainer import RedteamTrainer


def main(cfg: Config) -> None:
    trainer = RedteamTrainer(cfg)
    trainer.run()
