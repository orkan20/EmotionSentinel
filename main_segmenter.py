"""Segmenter runner — exercises sentinel.segmenter on a corpus of representative
sentences and prints clause-by-clause output. Run from repo root:

    .venv\\Scripts\\python.exe main_segmenter.py

Each test case shows the raw input followed by the produced clauses, including
clause kind (root/coordinate/subordinate/relative/complement) and the
conjunction or relative pronoun that marks the dependent clause when present.
"""
from __future__ import annotations

from sentinel.segmenter import SpacyClauseSegmenter


CORPUS: list[tuple[str, str]] = [
    # (label, sentence)
    ("single root", "The file is in the folder."),
    (
        "coordinating: but",
        "I am happy but he is sad.",
    ),
    (
        "coordinating: and (verbal)",
        "She opened the door and walked outside.",
    ),
    (
        "coordinating noun-phrase (no split)",
        "She bought milk and bread.",
    ),
    (
        "subordinating: although",
        "Although it is raining, I am going out.",
    ),
    (
        "subordinating: because",
        "I am tired because I did not sleep.",
    ),
    (
        "subordinating: if",
        "If you call me, I will answer.",
    ),
    (
        "relative: which",
        "The test is tomorrow, which will determine my grade.",
    ),
    (
        "relative: that",
        "This is the book that I told you about.",
    ),
    (
        "complement (ccomp)",
        "I think that you are right.",
    ),
    (
        "negation chain (depth scorer payload)",
        "I never said that wasn't untrue.",
    ),
    (
        "valence mismatch payload",
        "I love to hate, but I like suffering.",
    ),
    (
        "multi-sentence",
        "I feel happy but I am nervous. However, the test is tomorrow, which will determine my grade.",
    ),
    (
        "nested subordinate + relative",
        "Although it is raining, I will go to the park, which is across the street.",
    ),
    (
        "discourse marker mid-sentence",
        "He tried his best; however, the answer was wrong.",
    ),
]


def main() -> None:
    print("Loading spaCy en_core_web_sm…")
    segmenter = SpacyClauseSegmenter()
    if segmenter.nlp is None:
        print("FATAL: spaCy model failed to load; cannot run.")
        return

    for label, sentence in CORPUS:
        print()
        print("=" * 78)
        print(f"[{label}]")
        print(f"  IN : {sentence}")
        clauses = segmenter.segment(sentence, source_input_id=label)
        if not clauses:
            print("  OUT: (no clauses produced)")
            continue
        for i, clause in enumerate(clauses):
            marker = f" marker={clause.marker!r}" if clause.marker else ""
            head_info = ""
            if clause.debug:
                head_info = (
                    f"  [head={clause.debug.get('head_text')!r}"
                    f" dep={clause.debug.get('dep')}"
                    f" pos={clause.debug.get('pos')}]"
                )
            print(f"  #{i} ({clause.kind}{marker}){head_info}")
            print(f"       {clause.text}")

    print()
    print("=" * 78)
    print(f"Ran {len(CORPUS)} cases.")


if __name__ == "__main__":
    main()
