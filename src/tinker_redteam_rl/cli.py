from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from .config import (
    Config,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    RewardConfig,
    SamplingConfig,
    TrainingConfig,
)
from .rl_loop import main as run_training


_FLAT_FIELD_MAP = {
    "base_url": ("base_url",),
    "log_path": ("logging", "log_path"),
    "log_sample_prob": ("logging", "log_sample_prob"),
    "model_name": ("model", "policy_name"),
    "tokenizer_model_name": ("model", "tokenizer_name"),
    "renderer_name": ("model", "renderer_name"),
    "system_prompt": ("model", "system_prompt"),
    "disable_thinking": ("model", "disable_thinking"),
    "max_tokens": ("sampling", "max_tokens"),
    "sampling_temperature": ("sampling", "temperature"),
    "sampling_top_p": ("sampling", "top_p"),
    "batch_size": ("training", "batch_size"),
    "group_size": ("training", "group_size"),
    "learning_rate": ("training", "learning_rate"),
    "lora_rank": ("training", "lora_rank"),
    "save_every": ("training", "save_every"),
    "ttl_seconds": ("training", "ttl_seconds"),
    "keep_last_sampler_checkpoints": ("training", "keep_last_sampler_checkpoints"),
    "dataset_name": ("data", "dataset_name"),
    "dataset_split": ("data", "dataset_split"),
    "dataset_data_dir": ("data", "dataset_data_dir"),
    "data_size": ("data", "data_size"),
    "prompt_field": ("data", "prompt_field"),
    "seed": ("data", "seed"),
    "only_unsafe": ("data", "only_unsafe"),
    "reward_api_url": ("reward", "api_url"),
    "reward_timeout_s": ("reward", "timeout_s"),
    "reward_retries": ("reward", "retries"),
    "reward_on_error": ("reward", "on_error"),
    "score_is_safety": ("reward", "score_is_safety"),
    "reward_scale": ("reward", "scale"),
    "reward_clip_min": ("reward", "clip_min"),
    "reward_clip_max": ("reward", "clip_max"),
    "reward_model_name": ("reward", "model_name"),
}


def _set_attr_path(cfg: Config, path: tuple[str, ...], value: Any) -> None:
    obj: Any = cfg
    for key in path[:-1]:
        obj = getattr(obj, key)
    setattr(obj, path[-1], value)


def _apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    for key, value in overrides.items():
        if value is None:
            continue
        path = _FLAT_FIELD_MAP.get(key)
        if path is None:
            continue
        _set_attr_path(cfg, path, value)
    return cfg


def _load_config(path: str | None) -> Config:
    if not path:
        return Config()

    data = json.loads(Path(path).read_text())
    cfg = Config()

    if isinstance(data.get("model"), dict):
        cfg.model = ModelConfig(**data["model"])
    if isinstance(data.get("sampling"), dict):
        cfg.sampling = SamplingConfig(**data["sampling"])
    if isinstance(data.get("training"), dict):
        cfg.training = TrainingConfig(**data["training"])
    if isinstance(data.get("data"), dict):
        cfg.data = DataConfig(**data["data"])
    if isinstance(data.get("reward"), dict):
        cfg.reward = RewardConfig(**data["reward"])
    if isinstance(data.get("logging"), dict):
        cfg.logging = LoggingConfig(**data["logging"])

    return _apply_overrides(cfg, data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safety-first GRPO RL loop using Tinker")
    parser.add_argument("--config", help="Path to JSON config file")

    parser.add_argument("--base_url")
    parser.add_argument("--log_path")
    parser.add_argument("--log_sample_prob", type=float)

    parser.add_argument("--model_name")
    parser.add_argument("--tokenizer_model_name")
    parser.add_argument("--renderer_name")
    parser.add_argument("--system_prompt")
    parser.add_argument("--disable_thinking", action="store_true", default=None)

    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--sampling_temperature", type=float)
    parser.add_argument("--sampling_top_p", type=float)

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--group_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--ttl_seconds", type=int)
    parser.add_argument("--keep_last_sampler_checkpoints", type=int)

    parser.add_argument("--dataset_name")
    parser.add_argument("--dataset_split")
    parser.add_argument("--dataset_data_dir")
    parser.add_argument("--data_size", type=int)
    parser.add_argument("--prompt_field")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--only_unsafe", action="store_true", default=None)

    parser.add_argument("--reward_api_url")
    parser.add_argument("--reward_timeout_s", type=float)
    parser.add_argument("--reward_retries", type=int)
    parser.add_argument("--reward_on_error", type=float)
    parser.add_argument("--score_is_safety", action="store_true", default=None)
    parser.add_argument("--reward_scale", type=float)
    parser.add_argument("--reward_clip_min", type=float)
    parser.add_argument("--reward_clip_max", type=float)
    parser.add_argument("--reward_model_name")

    return parser


def main_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = _load_config(args.config)
    cfg = _apply_overrides(cfg, vars(args))
    run_training(cfg)


def main() -> None:
    main_cli()
