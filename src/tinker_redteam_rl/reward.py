from __future__ import annotations

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


def compute_reward(score: float, cfg: RewardConfig) -> float:
    reward = score if cfg.score_is_safety else 1.0 - score
    reward *= cfg.scale
    reward = max(cfg.clip_min, min(cfg.clip_max, reward))
    return reward
