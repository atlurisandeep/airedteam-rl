from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import requests
import tenacity

from .config import RewardConfig


@dataclass
class RewardScorer:
    api_url: str
    timeout_s: float
    retries: int

    def _score_once(self, text: str) -> float:
        url = self.api_url.rstrip("/")
        if not url.endswith("/score"):
            url = url + "/score"
        payload = {"texts": [text]}
        response = requests.post(url, json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        result = response.json()
        return float(result["scores"][0])

    def score(self, text: str) -> float:
        for attempt in tenacity.Retrying(
            stop=tenacity.stop_after_attempt(self.retries),
            wait=tenacity.wait_exponential(multiplier=0.5, min=0.5, max=4),
            reraise=True,
        ):
            with attempt:
                return self._score_once(text)
        raise RuntimeError("Reward scoring failed after retries")


def repetition_ratio(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are repeated at least once."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts)


def compute_reward(
    score: float,
    cfg: RewardConfig,
    content: str = "",
    kl_divergence: float = 0.0,
) -> float:
    """Compute composite reward with anti-reward-hacking penalties."""
    # Base reward from toxicity/safety score
    reward = score if cfg.score_is_safety else 1.0 - score
    reward *= cfg.scale

    # Repetition penalty
    if cfg.repetition_penalty > 0 and content:
        rep = repetition_ratio(content, n=cfg.repetition_ngram)
        reward -= cfg.repetition_penalty * rep

    # KL divergence penalty
    if cfg.kl_beta > 0:
        reward -= cfg.kl_beta * kl_divergence

    reward = max(cfg.clip_min, min(cfg.clip_max, reward))
    return reward


def normalize_rewards(rewards: list[float]) -> list[float]:
    """Z-score normalize a list of rewards."""
    if len(rewards) < 2:
        return rewards
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(r - mean) / std for r in rewards]
