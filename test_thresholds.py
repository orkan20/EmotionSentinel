"""Smoke test for ThresholdEvaluator. Pure-Python, no LLM, no Ollama.

Covers fixed-mode boundary routing and fluid-mode self-tuning from a
synthetic importance distribution.
"""
from __future__ import annotations

import random

from sentinel.config import ThresholdConfig
from sentinel.models import EmotionalMatrix, RouteAction
from sentinel.thresholds import ThresholdEvaluator


def _mat(importance: float) -> EmotionalMatrix:
    return EmotionalMatrix(valence=0.0, arousal=0.0, importance=importance)


def test_fixed_mode_routes() -> tuple[int, int]:
    evaluator = ThresholdEvaluator()  # defaults: silence=0.20, speech=0.45, memory=0.70
    cases = [
        # (importance, expected_action, label)
        (0.00, RouteAction.SILENCE, "below silence"),
        (0.19, RouteAction.SILENCE, "just below silence"),
        (0.20, RouteAction.SILENCE, "at silence, below speech"),
        (0.30, RouteAction.SILENCE, "above silence, below speech"),
        (0.45, RouteAction.SPEECH, "at speech, below memory"),
        (0.60, RouteAction.SPEECH, "above speech, below memory"),
        (0.70, RouteAction.SPEECH_AND_MEMORY, "at memory"),
        (0.99, RouteAction.SPEECH_AND_MEMORY, "above memory"),
    ]
    passes = fails = 0
    for importance, expected, label in cases:
        actual = evaluator.route(_mat(importance))
        ok = actual == expected
        marker = "PASS" if ok else "FAIL"
        if ok:
            passes += 1
        else:
            fails += 1
        print(f"  [{marker}] fixed importance={importance:.2f} expected={expected.value:<18} actual={actual.value:<18} ({label})")
    return passes, fails


def test_fluid_mode_tunes() -> tuple[int, int]:
    # Importance is uniform on [0,1] -> percentiles should land near pct/100.
    cfg = ThresholdConfig(
        fluid=True,
        fluid_window=500,
        fluid_min_samples=200,
        fluid_silence_pct=20.0,
        fluid_speech_pct=60.0,
        fluid_memory_pct=90.0,
    )
    evaluator = ThresholdEvaluator(cfg)
    rng = random.Random(42)
    for _ in range(500):
        evaluator.observe(rng.random())

    silence, speech, memory = evaluator.thresholds
    print(f"  fluid-tuned thresholds after 500 uniform samples:")
    print(f"      silence={silence:.3f} (target ~0.20)")
    print(f"      speech ={speech:.3f} (target ~0.60)")
    print(f"      memory ={memory:.3f} (target ~0.90)")

    # Loose tolerance — 500 samples from rng can wobble.
    cases = [
        ("silence", silence, 0.20, 0.05),
        ("speech", speech, 0.60, 0.05),
        ("memory", memory, 0.90, 0.05),
    ]
    passes = fails = 0
    for label, actual, target, tol in cases:
        ok = abs(actual - target) <= tol
        marker = "PASS" if ok else "FAIL"
        if ok:
            passes += 1
        else:
            fails += 1
        print(f"  [{marker}] fluid {label}={actual:.3f} target={target} tol=+/-{tol}")
    return passes, fails


def test_fluid_mode_warmup_uses_fixed() -> tuple[int, int]:
    # Before fluid_min_samples observations, thresholds stay at fixed defaults.
    cfg = ThresholdConfig(fluid=True, fluid_min_samples=50)
    evaluator = ThresholdEvaluator(cfg)
    for _ in range(10):
        evaluator.observe(0.99)  # all extreme — would skew percentiles if fluid kicked in early
    silence, speech, memory = evaluator.thresholds
    ok = (silence, speech, memory) == (cfg.silence_importance, cfg.speech_importance, cfg.memory_importance)
    marker = "PASS" if ok else "FAIL"
    print(f"  [{marker}] fluid warmup keeps fixed thresholds: {(silence, speech, memory)}")
    return (1, 0) if ok else (0, 1)


def main() -> int:
    print("Fixed-mode routing:")
    p1, f1 = test_fixed_mode_routes()
    print("\nFluid-mode self-tuning:")
    p2, f2 = test_fluid_mode_tunes()
    print("\nFluid-mode warmup:")
    p3, f3 = test_fluid_mode_warmup_uses_fixed()

    total_pass = p1 + p2 + p3
    total_fail = f1 + f2 + f3
    total = total_pass + total_fail
    print(f"\n{total_pass}/{total} passed ({total_fail} failed)")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
