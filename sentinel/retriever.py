# TODO: retrieval not yet wired into DocumentMatrix.memories (see docs/decisions.md)
from __future__ import annotations

from datetime import datetime, timezone

from sentinel.config import RetrievalConfig
from sentinel.embeddings import EmbeddingModel, cosine_similarity
from sentinel.memory_store import SQLiteMemoryStore
from sentinel.models import Memory


class MemoryRetriever:
    def __init__(
        self,
        store: SQLiteMemoryStore,
        embedding_model: EmbeddingModel,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.config = config or RetrievalConfig()

    def retrieve(self, query_text: str) -> list[Memory]:
        query_embedding = self.embedding_model.embed(query_text)
        memories = self.store.list_memories()
        ranked = sorted(
            memories,
            key=lambda memory: self._score(memory, query_embedding),
            reverse=True,
        )
        selected = ranked[: self.config.limit]
        self.store.mark_accessed([memory.id for memory in selected])
        return selected

    def _score(self, memory: Memory, query_embedding: list[float]) -> float:
        similarity = max(0.0, cosine_similarity(query_embedding, memory.embedding))
        importance = memory.importance
        recency = self._recency_score(memory.created_at)
        return (
            similarity * self.config.similarity_weight
            + importance * self.config.importance_weight
            + recency * self.config.recency_weight
        )

    @staticmethod
    def _recency_score(created_at: str) -> float:
        try:
            created = datetime.fromisoformat(created_at)
        except ValueError:
            return 0.0
        age_seconds = max(
            0.0, (datetime.now(timezone.utc) - created).total_seconds()
        )
        return 1.0 / (1.0 + age_seconds / 86400.0)
