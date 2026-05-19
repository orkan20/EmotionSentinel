# Minimal sentinel package initialization
# Only exports core model classes; avoids circular imports from pipeline/segmenter

from .models import (
    Clause,
    DepthScore,
    EmotionalMatrix,
    ProcessedClause,
    Memory,
    VoiceDirective,
    SentinelResult,
    Sender,
    RouteAction,
)

__all__ = [
    "Clause",
    "DepthScore",
    "EmotionalMatrix",
    "ProcessedClause",
    "Memory",
    "VoiceDirective",
    "SentinelResult",
    "Sender",
    "RouteAction",
]
