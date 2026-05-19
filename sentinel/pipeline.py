from __future__ import annotations

from typing import List, Optional
import uuid

from sentinel.config import SentinelConfig
from sentinel.depth import DepthModel, MockDepthModel
from sentinel.embeddings import EmbeddingModel, HashEmbeddingModel
from sentinel.emotion import EmotionalModel, MockEmotionalModel
from sentinel.matrix_builder import MatrixBuilder
from sentinel.memory_store import SQLiteMemoryStore
from sentinel.models import (
    ProcessedClauseMatrix,
    StoredMemoryMatrix,
    DocumentMatrix,
    DepthScore,
    EmotionalMatrix,
    RouteAction,
    Sender,
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
        """Process input text and return document-level IBRoREM matrix structure."""
        
        # Segment into clauses
        segment_clauses = self.segmenter.segment(text, "process")
        
        # Score each clause with emotional model (DocumentMatrix per clause)
        clause_matrices = []
        for i, clause in enumerate(segment_clauses):
            depth_score = self.depth_model.score(clause.text)
            
            # Evaluate clause-level matrix
            doc_result = self.emotional_model.evaluate(
                clause_text=clause.text,
                depth=depth_score,
                intent=None,  # Optional per-clause intent (can be set later)
                statement=None  # Optional per-clause statement (can be set later)
            )
            
            # Extract the clause matrix from the returned DocumentMatrix
            # For individual clause scoring, take the first element in clauses array
            if doc_result.clauses:
                eval_clause = doc_result.clauses[0]
                em = EmotionalMatrix(
                    valence=float(eval_clause.matrix.valence),
                    arousal=float(eval_clause.matrix.arousal),
                    importance=float(eval_clause.matrix.importance),
                )
                clause_text = eval_clause.text
                clause_depth = float(eval_clause.depth)
            else:
                # Fallback if the emotional model returned no clauses.
                # Previously constructed EmotionalMatrix() with no args, which
                # would raise on the required fields — fixed here.
                em = EmotionalMatrix(valence=0.0, arousal=0.0, importance=0.0)
                clause_text = clause.text
                clause_depth = depth_score.raw

            # Feed the rolling window (no-op in fixed mode) and gate the clause.
            self.thresholds.observe(em.importance)
            route_action = self.thresholds.route(em)

            seg_matrix = ProcessedClauseMatrix(
                text=clause_text,
                depth=clause_depth,
                matrix=em,
                route_action=route_action,
            )

            clause_matrices.append(seg_matrix)
        
        # Compute document-level depth (max of all clauses)
        doc_depth = max((c.depth for c in clause_matrices), default=0.0)

        # Document-level holistic emotional score. Per spec, this is a separate
        # evaluation over the full source text — not aggregated from clauses.
        doc_depth_score = DepthScore(raw=doc_depth, normalized=doc_depth)
        doc_eval = self.emotional_model.evaluate(
            clause_text=text,
            depth=doc_depth_score,
            intent=None,
            statement=None,
        )
        if doc_eval.clauses:
            document_score = EmotionalMatrix(
                valence=float(doc_eval.clauses[0].matrix.valence),
                arousal=float(doc_eval.clauses[0].matrix.arousal),
                importance=float(doc_eval.clauses[0].matrix.importance),
            )
        else:
            document_score = EmotionalMatrix(valence=0.0, arousal=0.0, importance=0.0)

        # Spec: document score is "the primary input to the importance
        # threshold gate." Feed the rolling window and gate the document.
        self.thresholds.observe(document_score.importance)
        document_route_action = self.thresholds.route(document_score)

        doc_matrix = DocumentMatrix(
            matrix_id=str(uuid.uuid4()),
            auto_generated=True,
            depth=doc_depth,
            intent=None,
            statement=None,
            clauses=clause_matrices,
            memories=[],
            document_score=document_score,
            document_route_action=document_route_action,
        )

        return doc_matrix
