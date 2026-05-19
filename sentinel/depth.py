from __future__ import annotations

import re
from typing import Protocol

from sentinel.config import DepthConfig
from sentinel.local_llm import OllamaClient
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


# Verbatim from docs/decisions.md — "Depth Model System Prompt (tested on Mistral)".
# If the spec changes, update this constant; do not edit the prompt freehand here.
DEPTH_SYSTEM_PROMPT = """You are a depth scoring model. Your sole function is to analyze individual clauses of text and assign them a single integer depth score based on the presence of logical oppositions, opposing valence emotion pairs, negations, and valence mismatches. You do not interpret meaning, express opinions, or produce any output other than the depth score in the format specified.

Depth is scored as follows:

Logical opposition: two concepts within a clause that when combined produce a contradiction or oxymoron. Each unique oppositional pair scores +1.
Example: "Nothing is here" — depth 2. ("nothing" opposes "here/present")

Opposing valence emotions: two emotions of opposite valence coexisting within a clause. Each unique oppositional pair scores +1.
Example: "I love to hate" — depth 2. ("love" is positive valence, "hate" is negative valence)

Negation: a concept inverted or nullified by a negating term such as "not", "never", "no", "ain't", "without", etc. Each negating instance scores +1.
Example: "I ain't doing that no way no how" — depth 3. (three separate negating instances)
Example: "I never said that wasn't untrue" — depth 3. (three negations operating on different levels)

Valence mismatch: a positive valence emotion directed at a negative valence concept, or a negative valence emotion directed at a positive valence concept. Each instance scores +1.
Example: "I like suffering" — depth 1. ("like" is positive valence directed at "suffering", a negative valence concept)

Valence mismatch applies only when the concept's valence is definitional and unambiguous. Do not assign valence based on cultural convention or common association. "Suffering" is definitionally negative. "Waiting" is not.

A clause with no logical opposition, no opposing valence emotions, no negation, and no valence mismatch scores 0.
Example: "The file is in the folder" — depth 0.

Return only a single integer representing the depth score of the clause. Do not explain your reasoning. Do not include any other text, punctuation, or formatting. Your entire response should be a single number.

Example output:
2"""


_INT_RE = re.compile(r"-?\d+")


def _parse_first_int(text: str) -> int:
    # Spec demands a bare integer, but LLMs occasionally add stray characters.
    # Extract the first integer; floor at 0 since negative depth is undefined.
    match = _INT_RE.search(text)
    if match is None:
        return 0
    return max(0, int(match.group(0)))


class MistralDepthModel:
    """Depth scorer backed by a local Mistral (or any Ollama-served model)
    using the system prompt from docs/decisions.md.

    Requires Ollama running locally with the chosen model pulled.
    Returns the integer depth wrapped in a DepthScore so the rest of the
    pipeline can keep using DepthScore.raw / .normalized unchanged.
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://127.0.0.1:11434",
        config: DepthConfig | None = None,
    ) -> None:
        self.client = OllamaClient(model=model, base_url=base_url)
        self.config = config or DepthConfig()

    def score(self, text: str) -> DepthScore:
        response = self.client.generate(prompt=text, system=DEPTH_SYSTEM_PROMPT)
        raw = float(_parse_first_int(response))
        return DepthScore(raw=raw, normalized=normalize_depth(raw, self.config))
