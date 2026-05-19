# TODO: Importance/Cessation Algorithm — not yet integrated into SentinelPipeline (see docs/decisions.md)
from __future__ import annotations

from sentinel.config import ThresholdConfig
from sentinel.models import EmotionalMatrix, RouteAction


class ThresholdEvaluator:
    def __init__(self, config: ThresholdConfig | None = None) -> None:
        self.config = config or ThresholdConfig()

    def route(self, matrix: EmotionalMatrix) -> RouteAction:
        should_speak = matrix.importance >= self.config.speech_importance
        should_remember = matrix.importance >= self.config.memory_importance

        if should_speak and should_remember:
            return RouteAction.SPEECH_AND_MEMORY
        if should_speak:
            return RouteAction.SPEECH
        if should_remember:
            return RouteAction.MEMORY
        return RouteAction.SILENCE
