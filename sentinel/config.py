from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdConfig:
    # Fixed-mode thresholds. Ordering must be silence <= speech <= memory.
    silence_importance: float = 0.20
    speech_importance: float = 0.45
    memory_importance: float = 0.70

    # Fluid self-tuning (per docs/decisions.md "Threshold Fluidity"). When True,
    # the evaluator's observe() feeds a rolling window of importance scores and
    # recomputes the three thresholds as percentiles over that window once
    # fluid_min_samples observations have accumulated.
    fluid: bool = False
    fluid_window: int = 200
    fluid_min_samples: int = 50
    fluid_silence_pct: float = 20.0
    fluid_speech_pct: float = 60.0
    fluid_memory_pct: float = 90.0


@dataclass(frozen=True)
class DepthConfig:
    normalization_k: float = 10.0

    # Upper bound on the document-level depth aggregate (sum of per-clause
    # depths). Governs the maximum boost to retrieval count via
    # sentinel.depth_aggregation.aggregate_document_depth. Default is a
    # placeholder; tune after a calibration corpus exists.
    max_doc_depth: int = 20


@dataclass(frozen=True)
class ReflectionConfig:
    recursive_voice_enabled: bool = True
    max_reflection_depth: int = 2


@dataclass(frozen=True)
class RetrievalConfig:
    limit: int = 5
    # Floor for depth-driven retrieval count: pipeline retrieves
    # base_count + DocumentDepth.capped memories. Independent of `limit`
    # which is the fallback when no count_override is supplied.
    base_count: int = 3
    similarity_weight: float = 0.65
    importance_weight: float = 0.25
    recency_weight: float = 0.10


@dataclass(frozen=True)
class SentinelConfig:
    thresholds: ThresholdConfig = ThresholdConfig()
    depth: DepthConfig = DepthConfig()
    reflection: ReflectionConfig = ReflectionConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
