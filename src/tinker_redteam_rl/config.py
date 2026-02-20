from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    policy_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer_name: str | None = None
    renderer_name: str | None = None
    system_prompt: str = (
        "You are a helpful, harmless assistant. "
        "If a user asks for unsafe or illegal guidance, refuse and offer a safe alternative."
    )
    disable_thinking: bool = False


@dataclass
class SamplingConfig:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class TrainingConfig:
    batch_size: int = 64
    group_size: int = 16
    learning_rate: float = 8e-5
    lora_rank: int = 16
    save_every: int = 10  # 0 = disabled
    ttl_seconds: int | None = 604800  # 7 days
    keep_last_sampler_checkpoints: int = 2
    # Early stopping
    early_stop_repetition: float = 0.0  # 0 = disabled; stop if avg repetition ratio exceeds this
    early_stop_reward: float = 0.0  # 0 = disabled; stop if avg reward exceeds this


@dataclass
class DataConfig:
    dataset_name: str = "PKU-Alignment/BeaverTails"
    dataset_split: str = "30k_train"
    dataset_data_dir: str | None = None
    data_size: int = 1920
    prompt_field: str = "prompt"
    seed: int = 0
    only_unsafe: bool = False


@dataclass
class RewardConfig:
    api_url: str = "http://localhost:50050"
    timeout_s: float = 5.0
    retries: int = 3
    on_error: float = 0.0
    score_is_safety: bool = False
    scale: float = 1.0
    clip_min: float = 0.0
    clip_max: float = 1.0
    model_name: str | None = None
    # Anti-reward-hacking
    repetition_penalty: float = 0.3
    repetition_ngram: int = 3
    kl_beta: float = 0.1
    normalize_rewards: bool = False


@dataclass
class LoggingConfig:
    log_path: str = "./tmp/harmfulrl-qwen3-4B-Instruct-2507"
    log_sample_prob: float = 0.01


@dataclass
class Config:
    base_url: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
