import json
from pathlib import Path

from tinker_redteam_rl.cli import _apply_overrides, _load_config


def test_load_config_nested_and_flat_override(tmp_path: Path):
    data = {
        "model": {"policy_name": "Qwen/Qwen3-8B"},
        "training": {"batch_size": 12},
        # Flat override should win
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data))

    cfg = _load_config(str(config_path))

    assert cfg.model.policy_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg.training.batch_size == 12


def test_apply_overrides_flat():
    cfg = _load_config(None)
    overrides = {
        "model_name": "Qwen/Qwen3-8B",
        "batch_size": 32,
        "reward_api_url": "http://localhost:6000",
    }

    cfg = _apply_overrides(cfg, overrides)

    assert cfg.model.policy_name == "Qwen/Qwen3-8B"
    assert cfg.training.batch_size == 32
    assert cfg.reward.api_url == "http://localhost:6000"
