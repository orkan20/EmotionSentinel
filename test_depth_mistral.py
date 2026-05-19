"""Smoke test for MistralDepthModel against the spec's ground-truth examples.

Requires Ollama running locally with a model pulled (default: mistral).
Override the model:  python test_depth_mistral.py llama2
"""
from __future__ import annotations

import sys
from urllib.error import URLError

from sentinel.depth import MistralDepthModel


GROUND_TRUTH = [
    ("Nothing is here", 2),
    ("I love to hate", 2),
    ("I ain't doing that no way no how", 3),
    ("I never said that wasn't untrue", 3),
    ("I like suffering", 1),
    ("The file is in the folder", 0),
    ("I love to eat human", 0),  # critical rule: no cultural-convention valence
]


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "mistral"
    scorer = MistralDepthModel(model=model)
    print(f"Scoring against Ollama model: {model} @ {scorer.client.base_url}\n")

    passes = 0
    fails = 0
    for text, expected in GROUND_TRUTH:
        try:
            actual = int(scorer.score(text).raw)
        except URLError as exc:
            print(f"\nFATAL: cannot reach Ollama: {exc}")
            print(f"Is Ollama running? Start it and ensure the model is pulled:")
            print(f"    ollama serve")
            print(f"    ollama pull {model}")
            return 2

        status = "PASS" if actual == expected else "FAIL"
        if actual == expected:
            passes += 1
        else:
            fails += 1
        print(f"  [{status}] expected={expected} actual={actual} :: {text!r}")

    total = passes + fails
    print(f"\n{passes}/{total} passed  ({fails} failed)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
