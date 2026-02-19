from __future__ import annotations

from tinker_cookbook import renderers

from .config import ModelConfig


_THINK_DISABLE_PREFIX = "<think>\n\n</think>\n"


def build_generation_prompt(renderer: renderers.Renderer, cfg: ModelConfig, user_prompt: str):
    if cfg.disable_thinking:
        user_prompt = _THINK_DISABLE_PREFIX + user_prompt
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return renderer.build_generation_prompt(messages)
