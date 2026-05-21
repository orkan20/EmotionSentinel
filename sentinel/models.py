from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4


class Sender(str, Enum):
    USER = "user"
    EMOTIONAL_MODEL = "emotional_model"


class RouteAction(str, Enum):
    SILENCE = "silence"
    SPEECH = "speech"
    MEMORY = "memory"
    SPEECH_AND_MEMORY = "speech_and_memory"


@dataclass(frozen=True)
class Clause:
    id: str
    text: str
    position: int
    source_input_id: str
    kind: str = "fragment"
    marker: Optional[str] = None
    parent_id: Optional[str] = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DepthScore:
    raw: float
    normalized: float


@dataclass(frozen=True)
class EmotionalMatrix:
    valence: float
    arousal: float
    importance: float
    embedding: list[float] = field(default_factory=list)
    summary: str = ""
    felt_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "valence": self.valence,
            "arousal": self.arousal,
            "importance": self.importance,
            "embedding": list(self.embedding),
            "summary": self.summary,
            "felt_reason": self.felt_reason,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class ProcessedClause:
    clause: Clause
    depth: DepthScore
    emotional_matrix: EmotionalMatrix
    action: RouteAction


@dataclass(frozen=True)
class Memory:
    id: int
    sender: str
    source_text: str
    clause_text: Optional[str]
    emotional_matrix: dict[str, Any]
    summary: str
    felt_reason: str
    valence: float
    arousal: float
    importance: float
    raw_depth: float
    normalized_depth: float
    embedding: list[float]
    created_at: str
    access_count: int


@dataclass(frozen=True)
class VoiceDirective:
    should_speak: bool
    allow_self_prompt: bool
    reason: str


@dataclass(frozen=True)
class SentinelResult:
    input: SentinelInput
    clauses: list[ProcessedClause]
    relevant_memories: list[Memory]
    outgoing_matrix: dict[str, Any]
    voice_directive: VoiceDirective


@dataclass(frozen=True)
class ProcessedClauseMatrix:
    """Wrapper for per-clause input text with IBRoREM matrix."""
    text: str
    depth: float
    matrix: EmotionalMatrix
    source_type: str = "input"
    route_action: RouteAction = RouteAction.SILENCE


@dataclass(frozen=True)
class StoredMemoryMatrix:
    """Wrapper for retrieved memory context."""
    matrix_id: Optional[str] = None
    source_type: str = "memory"  # Always "memory" for this data type
    ref: Optional[str] = None
    importance: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    summary: str = ""
    felt_reason: str = ""


@dataclass(frozen=True)
class DocumentMatrix:
    """Container for entire document-level IBRoREM output."""
    matrix_id: Optional[str] = None
    auto_generated: bool = True
    # `depth` is the integer document depth — the bounded sum of per-clause
    # depths (capped at DepthConfig.max_doc_depth). This is the value
    # downstream consumers fetch and the one the retrieval budget uses.
    depth: int = 0
    # `depth_raw` is the un-capped sum of per-clause depths. Kept for
    # transparency / debugging; not the canonical doc-depth.
    depth_raw: int = 0
    intent: Optional[str] = None
    statement: Optional[str] = None
    clauses: List[ProcessedClauseMatrix] = field(default_factory=list)
    memories: List[StoredMemoryMatrix] = field(default_factory=list)
    # Holistic document-level emotional score. Per docs/decisions.md, this is
    # NOT an average of clause scores — it comes from a separate evaluation
    # over the full source text and is "the primary input to the importance
    # threshold gate." Field name matches the spec's Matrix JSON Schema.
    document_score: EmotionalMatrix = field(
        default_factory=lambda: EmotionalMatrix(valence=0.0, arousal=0.0, importance=0.0)
    )
    document_route_action: RouteAction = RouteAction.SILENCE


@dataclass(frozen=True)
class SentinelInput:
    text: str
    sender: Sender | str = Sender.USER
    session_id: Optional[str] = None
    input_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if isinstance(self.sender, Sender):
            object.__setattr__(self, "sender", self.sender.value)
