from __future__ import annotations

from typing import Optional
import uuid

from sentinel.config import SentinelConfig
from sentinel.depth import DepthModel, MockDepthModel
from sentinel.depth_aggregation import aggregate_document_depth
from sentinel.embeddings import EmbeddingModel, HashEmbeddingModel
from sentinel.emotion import EmotionalModel, MockEmotionalModel
from sentinel.matrix_builder import MatrixBuilder
from sentinel.memory_store import SQLiteMemoryStore
from sentinel.models import (
    ProcessedClause,
    ProcessedClauseMatrix,
    StoredMemoryMatrix,
    DocumentMatrix,
    DepthScore,
    EmotionalMatrix,
    Memory,
    RouteAction,
    Sender,
    SentinelInput,
)
from sentinel.retriever import MemoryRetriever
from sentinel.segmenter import ClauseSegmenter, SpacyClauseSegmenter
from sentinel.thresholds import ThresholdEvaluator


class SentinelPipeline:
    def __init__(
        self,
        config: Optional[SentinelConfig] = None,
        segmenter: Optional[ClauseSegmenter] = None,
        depth_model: Optional[DepthModel] = None,
        emotional_model: Optional[EmotionalModel] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        memory_store: Optional[SQLiteMemoryStore] = None,
        matrix_builder: Optional[MatrixBuilder] = None,
    ) -> None:
        self.config = config or SentinelConfig()
        self.segmenter = segmenter or SpacyClauseSegmenter()
        self.depth_model = depth_model or MockDepthModel(self.config.depth)
        self.emotional_model = emotional_model or MockEmotionalModel()
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self.memory_store = memory_store or SQLiteMemoryStore()
        self.thresholds = ThresholdEvaluator(self.config.thresholds)
        self.retriever = MemoryRetriever(
            self.memory_store, self.embedding_model, self.config.retrieval
        )
        self.matrix_builder = matrix_builder or MatrixBuilder()

    def process(
        self,
        text: str,
        sender: Sender | str = Sender.USER,
        session_id: Optional[str] = None,
        reflection_depth: int = 0,
    ) -> DocumentMatrix:
        """Process input text and return document-level IBRoREM matrix structure.

        Orchestration order (depth-first):
          1. Build SentinelInput.
          2. Segment into clauses.
          3. Score per-clause depth (cheap, no emotional model yet).
          4. Aggregate into DocumentDepth (sum, capped).
          5. Retrieve memories sized by ``base_count + doc_depth.capped``.
          6. Run emotional model per clause and a separate document gestalt
             pass; embed each matrix.
          7. Threshold-gate per clause; write memories whose route_action
             clears the memory threshold.
          8. Threshold-gate the document; assemble the DocumentMatrix.
        """

        sentinel_input = SentinelInput(
            text=text, sender=sender, session_id=session_id
        )

        # 2. Segment.
        segment_clauses = self.segmenter.segment(text, "process")

        # 3. Per-clause depth — structural-only signal, no emotional model.
        clause_depth_scores: list[DepthScore] = [
            self.depth_model.score(clause.text) for clause in segment_clauses
        ]

        # 4. Aggregate to bounded integer document depth.
        doc_depth = aggregate_document_depth(
            (ds.raw for ds in clause_depth_scores), self.config.depth
        )

        # 5. Retrieve sized by document depth. Per design doc §2A/§5, doc depth
        # is the retrieval budget — deeper text pulls more memories.
        retrieval_count = self.config.retrieval.base_count + doc_depth.capped
        retrieved = self.retriever.retrieve(text, count_override=retrieval_count)
        memory_matrices = [_memory_to_stored(m) for m in retrieved]

        # 6 + 7. Per-clause emotional scoring, embedding, gating, memory write.
        clause_matrices: list[ProcessedClauseMatrix] = []
        for clause, depth_score in zip(segment_clauses, clause_depth_scores):
            doc_result = self.emotional_model.evaluate(
                clause_text=clause.text,
                depth=depth_score,
                intent=None,
                statement=None,
            )

            if doc_result.clauses:
                eval_clause = doc_result.clauses[0]
                clause_text = eval_clause.text
                clause_depth = float(eval_clause.depth)
                em = EmotionalMatrix(
                    valence=_clamp(float(eval_clause.matrix.valence), -1.0, 1.0),
                    arousal=_clamp(float(eval_clause.matrix.arousal), 0.0, 1.0),
                    importance=_clamp(float(eval_clause.matrix.importance), 0.0, 1.0),
                    embedding=self.embedding_model.embed(eval_clause.text),
                    summary=eval_clause.matrix.summary,
                    felt_reason=eval_clause.matrix.felt_reason,
                )
            else:
                clause_text = clause.text
                clause_depth = depth_score.raw
                em = EmotionalMatrix(
                    valence=0.0,
                    arousal=0.0,
                    importance=0.0,
                    embedding=self.embedding_model.embed(clause.text),
                )

            self.thresholds.observe(em.importance)
            route_action = self.thresholds.route(em)

            seg_matrix = ProcessedClauseMatrix(
                text=clause_text,
                depth=clause_depth,
                matrix=em,
                route_action=route_action,
            )
            clause_matrices.append(seg_matrix)

            # Write authority: only clauses clearing the memory threshold.
            if route_action in (RouteAction.MEMORY, RouteAction.SPEECH_AND_MEMORY):
                processed = ProcessedClause(
                    clause=clause,
                    depth=depth_score,
                    emotional_matrix=em,
                    action=route_action,
                )
                self.memory_store.write_memory(
                    sentinel_input=sentinel_input,
                    processed_clause=processed,
                    embedding=em.embedding,
                )

        # 8. Document gestalt pass. Per spec, a separate evaluation over the
        # full source text — not aggregated from clauses. Depth flows in as
        # context but does not modulate the resulting matrix (decorrelation
        # principle, design doc §6).
        doc_depth_score_for_model = DepthScore(
            raw=float(doc_depth.raw_sum),
            normalized=float(doc_depth.capped) / max(1, self.config.depth.max_doc_depth),
        )
        doc_eval = self.emotional_model.evaluate(
            clause_text=text,
            depth=doc_depth_score_for_model,
            intent=None,
            statement=None,
        )
        if doc_eval.clauses:
            document_score = EmotionalMatrix(
                valence=_clamp(float(doc_eval.clauses[0].matrix.valence), -1.0, 1.0),
                arousal=_clamp(float(doc_eval.clauses[0].matrix.arousal), 0.0, 1.0),
                importance=_clamp(float(doc_eval.clauses[0].matrix.importance), 0.0, 1.0),
                embedding=self.embedding_model.embed(text),
                summary=doc_eval.clauses[0].matrix.summary,
                felt_reason=doc_eval.clauses[0].matrix.felt_reason,
            )
        else:
            document_score = EmotionalMatrix(
                valence=0.0,
                arousal=0.0,
                importance=0.0,
                embedding=self.embedding_model.embed(text),
            )

        self.thresholds.observe(document_score.importance)
        document_route_action = self.thresholds.route(document_score)

        return DocumentMatrix(
            matrix_id=str(uuid.uuid4()),
            auto_generated=True,
            depth=doc_depth.capped,
            depth_raw=doc_depth.raw_sum,
            intent=None,
            statement=None,
            clauses=clause_matrices,
            memories=memory_matrices,
            document_score=document_score,
            document_route_action=document_route_action,
        )


def _memory_to_stored(memory: Memory) -> StoredMemoryMatrix:
    """Adapt a persisted Memory row to the StoredMemoryMatrix shape used in
    DocumentMatrix.memories. Keeps the heavy fields (embedding, source_text)
    off the document; downstream code can re-query memory_store by matrix_id
    if it needs more detail."""
    return StoredMemoryMatrix(
        matrix_id=str(memory.id),
        ref=memory.clause_text,
        importance=memory.importance,
        valence=memory.valence,
        arousal=memory.arousal,
        summary=memory.summary,
        felt_reason=memory.felt_reason,
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
