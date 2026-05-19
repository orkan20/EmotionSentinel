from __future__ import annotations

from typing import Protocol, Optional
import json
import re
from uuid import uuid4
import random
from datetime import datetime, timezone

from sentinel.models import (
    DepthScore, 
    EmotionalMatrix, 
    DocumentMatrix,
    ProcessedClauseMatrix,
    StoredMemoryMatrix
)


class EmotionalModel(Protocol):
    """Interface for emotional scoring of input text with IBRoREM matrices."""
    
    def evaluate(self, 
                 clause_text: str, 
                 depth: DepthScore,
                 intent: Optional[str] = None,
                 statement: Optional[str] = None) -> DocumentMatrix:
        """Evaluate a single clause and/or document for emotional coordinates.
        
        Args:
            clause_text: Text to score (per-clause or full text depending on usage)
            depth: Pre-computed depth score from DepthModel
            intent: Optional semantic direction (help|clarify|console|support|null)
            statement: Optional short gist for voice output
            
        Returns:
            DocumentMatrix containing:
                - clauses[]: per-clause matrices with IBRoREM coordinates
                - memories[]: retrieved memory context matrices
                - intent/statement: optional document-level steering fields
        """
        ...


class MockEmotionalModel(EmotionalModel):
    """Rule-based placeholder that returns IBRoREM matrices.
    
    Implements dual-layer scoring:
        1. Clause-level matrices for input text segments
        2. Optional memory context (mock retrieval) for demonstration
    
    Note: Real implementations should use trained models (e.g., EmoLLMs) 
          with fine-tuning on IBRoREM coordinates + intent/statement detection.
    """
    
    NEGATIVE_TERMS = {"sad", "afraid", "angry", "hurt", "alone", "abandoned"}
    POSITIVE_TERMS = {"happy", "safe", "love", "proud", "excited", "calm"}
    HIGH_AROUSAL_TERMS = {"urgent", "panic", "terrified", "furious", "excited"}
    
    VALID_INTENTS = {"help", "clarify", "console", "support", "inform", None}
    
    def __init__(self):
        self._memo_counter = 0
    
    def evaluate(self, 
                 clause_text: str, 
                 depth: DepthScore,
                 intent: Optional[str] = None,
                 statement: Optional[str] = None) -> DocumentMatrix:
        """Mock implementation for demonstration. Replace with trained model."""
        
        # Split into clauses if single text provided (fallback)
        words = re.findall(r"\w+", clause_text.lower())
        raw_importance = max(0.0, len(words) / 3.0)
        normalized_importance = raw_importance / (raw_importance + 10.0)
        
        negative_hits = sum(term in clause_text.lower() for term in self.NEGATIVE_TERMS)
        positive_hits = sum(term in clause_text.lower() for term in self.POSITIVE_TERMS)
        arousal_hits = sum(term in clause_text.lower() for term in self.HIGH_AROUSAL_TERMS)
        
        valence = max(-1.0, min(1.0, (positive_hits - negative_hits) * 0.35))
        arousal = max(0.05, min(1.0, 0.20 + arousal_hits * 0.25 + depth.normalized * 0.40))
        
        # Use intent if provided, otherwise pick from valid set or null
        final_intent = intent if intent in self.VALID_INTENTS else None
        
        # Generate simple statement based on sentiment
        statement_options = {
            "console": "Acknowledge their feelings and offer support.",
            "clarify": "Break down the concept into clear parts.",
            "support": "Encourage them to try this step by step.",
            "inform": "Provide relevant facts or instructions.",
            None: None  # No statement when intent is null
        }
        
        final_statement = (
            statement_options.get(final_intent) 
            if final_intent is not None else None
        )
        
        # Mock clause splitting for demonstration
        clauses_text_split = [clause_text]  # Fallback to single text
        
        matrix_clauses = []
        for idx, segment in enumerate(clauses_text_split):
            seg_words = re.findall(r"\w+", segment.lower())
            seg_raw_importance = max(0.0, len(seg_words) / 3.0)
            seg_matrix = EmotionalMatrix(
                valence=valence + random.uniform(-0.1, 0.1),
                arousal=arousal + random.uniform(-0.05, 0.05),
                importance=seg_raw_importance * 0.8,
                summary=segment[:240] if len(segment) <= 240 else segment[:200] + "...",
                felt_reason=f"Mock matrix: valence from sentiment keywords, arousal from energy words"
            )
            
            matrix_clauses.append(ProcessedClauseMatrix(
                text=segment.strip(),
                depth=float(depth.raw),
                matrix=seg_matrix
            ))
        
        # Mock memory retrieval (empty list in production)
        memories = []
        
        return DocumentMatrix(
            matrix_id=str(uuid4()),
            auto_generated=True,
            depth=float(depth.raw),
            intent=final_intent,
            statement=final_statement,
            clauses=matrix_clauses,
            memories=memories
        )


class EmollamaEmotionalModel(EmotionalModel):
    """Real emotional scorer using lzw1008/Emollama-chat-7b from Hugging Face.

    Loads tokenizer + model lazily on first evaluate() call so importing this
    module doesn't require transformers/torch at install time — only at use.
    First call downloads ~14GB; subsequent calls reuse the loaded model.
    Returns a DocumentMatrix wrapping a single ProcessedClauseMatrix scored
    from the LLM's JSON output, clamped to the spec's ranges. Parse failures
    yield zero matrices with a felt_reason explaining what went wrong, so the
    pipeline keeps moving instead of crashing.
    """

    DEFAULT_MODEL = "lzw1008/Emollama-chat-7b"
    SYSTEM_PROMPT = (
        "You are an emotional scoring model. Score text on valence, arousal, "
        "and importance."
    )
    USER_TEMPLATE = (
        "Score the following text on valence (-1.0 to 1.0), arousal (0.0 to 1.0), "
        "and importance (0.0 to 1.0). Return only a JSON object with those three "
        "fields and nothing else.\nText: {text}"
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 200,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Defer transformers/torch imports so module import doesn't require them.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto",
        )

    def _generate(self, clause_text: str) -> str:
        self._ensure_loaded()
        user = self.USER_TEMPLATE.format(text=clause_text)
        prompt = f"[INST] <<SYS>>\n{self.SYSTEM_PROMPT}\n<</SYS>>\n\n{user} [/INST]"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip echoed prompt; keep only the post-[/INST] response.
        if "[/INST]" in decoded:
            decoded = decoded.split("[/INST]", 1)[1]
        return decoded.strip()

    def _parse_scores(self, response: str) -> tuple[float, float, float, str]:
        # Returns (valence, arousal, importance, felt_reason). Defaults to
        # zeros + a diagnostic reason if the LLM didn't emit clean JSON.
        match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if match is None:
            return 0.0, 0.0, 0.0, f"emollama parse failed: no JSON object in {response[:120]!r}"
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return 0.0, 0.0, 0.0, f"emollama parse failed: {exc}; raw={match.group(0)[:120]!r}"
        valence = _clamp(_as_float(data.get("valence"), 0.0), -1.0, 1.0)
        arousal = _clamp(_as_float(data.get("arousal"), 0.0), 0.0, 1.0)
        importance = _clamp(_as_float(data.get("importance"), 0.0), 0.0, 1.0)
        return valence, arousal, importance, "scored by Emollama-chat-7b"

    def evaluate(
        self,
        clause_text: str,
        depth: DepthScore,
        intent: Optional[str] = None,
        statement: Optional[str] = None,
    ) -> DocumentMatrix:
        response = self._generate(clause_text)
        valence, arousal, importance, reason = self._parse_scores(response)

        em = EmotionalMatrix(
            valence=valence,
            arousal=arousal,
            importance=importance,
            summary=clause_text[:240],
            felt_reason=reason,
        )
        seg = ProcessedClauseMatrix(
            text=clause_text.strip(),
            depth=float(depth.raw),
            matrix=em,
        )
        return DocumentMatrix(
            matrix_id=str(uuid4()),
            auto_generated=True,
            depth=float(depth.raw),
            intent=intent,
            statement=statement,
            clauses=[seg],
            memories=[],
        )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _as_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
