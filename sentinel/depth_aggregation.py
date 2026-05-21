"""Document-level depth aggregation.

Aggregates per-clause integer depth scores into a single document depth that
is used as the **retrieval budget** for the memory layer (see EmotionSentinel
design doc §2A, §5: "Depth at the document level governs how many memories
are retrieved for that context").

The aggregation is a bounded integer sum:

  * Each per-clause depth is rounded to the nearest non-negative integer.
  * Those integers are summed into ``raw_sum``.
  * ``capped`` is ``min(raw_sum, config.max_doc_depth)``.

Depth is intentionally **decorrelated** from the emotional matrix per the
design doc §6 ("Valence/Depth Decorrelation"). This module does not modulate
valence/arousal/importance; depth flows separately into retrieval count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sentinel.config import DepthConfig


@dataclass(frozen=True)
class DocumentDepth:
    raw_sum: int
    capped: int


def aggregate_document_depth(
    clause_depths: Iterable[float], config: DepthConfig
) -> DocumentDepth:
    """Sum per-clause depths into a bounded integer document depth.

    Per-clause depths from the real depth model are integers; the mock model
    emits floats (``len(words) / 3.0``). Both are normalized to non-negative
    integers via ``round(max(0.0, d))`` before summing so the aggregate is
    always an ``int`` — exactly what the retrieval budget consumer needs.
    """
    raw_sum = sum(int(round(max(0.0, float(d)))) for d in clause_depths)
    return DocumentDepth(
        raw_sum=raw_sum, capped=min(raw_sum, config.max_doc_depth)
    )
