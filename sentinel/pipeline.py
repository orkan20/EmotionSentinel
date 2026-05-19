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
                seg_matrix = ProcessedClauseMatrix(
                    text=doc_result.clauses[0].text,
                    depth=float(doc_result.clauses[0].depth),
                    matrix=EmotionalMatrix(
                        valence=float(doc_result.clauses[0].matrix.valence),
                        arousal=float(doc_result.clauses[0].matrix.arousal),
                        importance=float(doc_result.clauses[0].matrix.importance)
                    )
                )
            else:
                # Fallback if empty (shouldn't happen in production)
                seg_matrix = ProcessedClauseMatrix(
                    text=clause.text,
                    depth=depth_score.raw,
                    matrix=EmotionalMatrix()
                )
            
            clause_matrices.append(seg_matrix)
        
        # Compute document-level depth (max of all clauses)
        doc_depth = max((c.depth for c in clause_matrices), default=0.0)
        
        # Build document matrix with intent/statement
        doc_matrix = DocumentMatrix(
            matrix_id=str(uuid.uuid4()),
            auto_generated=True,
            depth=doc_depth,
            intent=None,  # Can be set based on text analysis later
            statement=None,  # Can be set based on text analysis later
            clauses=clause_matrices,
            memories=[]  # Empty for demo; would load from database in production
        )
        
        return doc_matrix
