from __future__ import annotations

from collections import deque
from typing import Optional

from sentinel.config import ThresholdConfig
from sentinel.models import EmotionalMatrix, RouteAction


class ThresholdEvaluator:
    """Three-gate threshold model per docs/decisions.md.

    Fixed mode (default): uses ThresholdConfig.{silence,speech,memory}_importance
    as the three gates. Importance below silence -> SILENCE. Anything above
    speech routes to SPEECH; anything above memory additionally routes to
    MEMORY. The combined state is SPEECH_AND_MEMORY.

    Fluid mode (config.fluid=True): observe() feeds importance scores into a
    rolling window; once fluid_min_samples have accumulated, the three
    thresholds are recomputed as percentiles over the window on every
    observation. Guards against drift in the emotional model's importance
    calibration during continuous training.
    """

    def __init__(self, config: Optional[ThresholdConfig] = None) -> None:
        self.config = config or ThresholdConfig()
        self._observations: deque[float] = deque(maxlen=self.config.fluid_window)
        # Effective thresholds — mutated by observe() in fluid mode.
        self._silence = self.config.silence_importance
        self._speech = self.config.speech_importance
        self._memory = self.config.memory_importance

    def observe(self, importance: float) -> None:
        if not self.config.fluid:
            return
        self._observations.append(importance)
        if len(self._observations) < self.config.fluid_min_samples:
            return
        sorted_obs = sorted(self._observations)
        self._silence = _percentile(sorted_obs, self.config.fluid_silence_pct)
        self._speech = _percentile(sorted_obs, self.config.fluid_speech_pct)
        self._memory = _percentile(sorted_obs, self.config.fluid_memory_pct)

    def route(self, matrix: EmotionalMatrix) -> RouteAction:
        importance = matrix.importance
        if importance < self._silence:
            return RouteAction.SILENCE
        should_speak = importance >= self._speech
        should_remember = importance >= self._memory
        if should_speak and should_remember:
            return RouteAction.SPEECH_AND_MEMORY
        if should_speak:
            return RouteAction.SPEECH
        if should_remember:
            # Only reachable if memory < speech, which violates the spec's
            # silence <= speech <= memory ordering. Preserve old behavior.
            return RouteAction.MEMORY
        return RouteAction.SILENCE

    @property
    def thresholds(self) -> tuple[float, float, float]:
        """Current effective (silence, speech, memory). In fluid mode, this
        reflects the latest percentile-tuned values."""
        return (self._silence, self._speech, self._memory)


def _percentile(sorted_values: list[float], pct: float) -> float:
    # Linear-interpolated percentile. sorted_values is ascending.
    if not sorted_values:
        return 0.0
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    n = len(sorted_values)
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])
