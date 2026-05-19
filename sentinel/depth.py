from __future__ import annotations

import re
from typing import Protocol

from sentinel.config import DepthConfig
from sentinel.models import DepthScore


class DepthModel(Protocol):
    def score(self, text: str) -> DepthScore:
        ...


def normalize_depth(raw_depth: float, config: DepthConfig) -> float:
    if raw_depth <= 0:
        return 0.0
    return raw_depth / (raw_depth + config.normalization_k)


class MockDepthModel:
    """Deterministic placeholder until the local depth LLM is wired in."""

    def __init__(self, config: DepthConfig | None = None) -> None:
        self.config = config or DepthConfig()

    def score(self, text: str) -> DepthScore:
        words = re.findall(r"\w+", text)
        raw = max(0.0, len(words) / 3.0)
        return DepthScore(raw=raw, normalized=normalize_depth(raw, self.config))
