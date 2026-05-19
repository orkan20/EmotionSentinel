from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdConfig:
    speech_importance: float = 0.45
    memory_importance: float = 0.70


@dataclass(frozen=True)
class DepthConfig:
    normalization_k: float = 10.0


@dataclass(frozen=True)
class ReflectionConfig:
    recursive_voice_enabled: bool = True
    max_reflection_depth: int = 2


@dataclass(frozen=True)
class RetrievalConfig:
    limit: int = 5
    similarity_weight: float = 0.65
    importance_weight: float = 0.25
    recency_weight: float = 0.10


@dataclass(frozen=True)
class SentinelConfig:
    thresholds: ThresholdConfig = ThresholdConfig()
    depth: DepthConfig = DepthConfig()
    reflection: ReflectionConfig = ReflectionConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
