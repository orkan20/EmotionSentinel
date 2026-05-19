"""Smoke test for EmollamaEmotionalModel.

WARNING: first run downloads ~14GB of model weights from Hugging Face;
subsequent runs load from disk in 30-60s. GPU strongly recommended.

Requires transformers + torch installed and a Hugging Face cache (~/.cache/
huggingface) with enough free space.

Not auto-run by any other test or by the pipeline default. Run manually:

    python test_emotion_emollama.py
"""
from __future__ import annotations

from sentinel.emotion import EmollamaEmotionalModel
from sentinel.models import DepthScore


# Each example pairs a clause with a qualitative expectation. No hard
# pass/fail — Emollama is stochastic and the spec doesn't pin numbers.
# Eyeball the printed output for sanity.
EXAMPLES = [
    ("This is the best day of my entire life.", "expect strong positive valence, moderate-high importance"),
    ("I am terrified of what comes next.",       "expect strong negative valence, high arousal"),
    ("The file is in the folder.",               "expect near-zero valence, low importance, low arousal"),
    ("I love to hate this stupid bug.",          "expect mixed/negative valence, elevated arousal"),
    ("Nothing matters anymore.",                 "expect strong negative valence, low arousal"),
]


def main() -> int:
    scorer = EmollamaEmotionalModel()
    # Pre-warm the model so the first iteration's load time doesn't pollute
    # the per-call timing.
    print(f"Loading {scorer.model_name} (first run downloads ~14GB)...")
    scorer._ensure_loaded()
    print("Loaded.\n")

    for text, expectation in EXAMPLES:
        depth = DepthScore(raw=0.0, normalized=0.0)
        result = scorer.evaluate(clause_text=text, depth=depth)
        m = result.clauses[0].matrix
        print(f"text:        {text!r}")
        print(f"expectation: {expectation}")
        print(f"  valence={m.valence:+.3f}  arousal={m.arousal:.3f}  importance={m.importance:.3f}")
        print(f"  reason:    {m.felt_reason}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
