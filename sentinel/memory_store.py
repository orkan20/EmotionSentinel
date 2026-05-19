from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentinel.models import Memory, ProcessedClause, SentinelInput


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender TEXT NOT NULL,
    session_id TEXT,
    source_input_id TEXT NOT NULL,
    source_text TEXT NOT NULL,
    clause_text TEXT,
    emotional_matrix_json TEXT NOT NULL,
    summary TEXT,
    felt_reason TEXT,
    valence REAL NOT NULL,
    arousal REAL NOT NULL,
    importance REAL NOT NULL,
    raw_depth REAL NOT NULL,
    normalized_depth REAL NOT NULL,
    embedding_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS training_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text TEXT NOT NULL,
    target_matrix_json TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    used_for_training INTEGER DEFAULT 0
);
"""


class SQLiteMemoryStore:
    def __init__(self, path: str | Path = "sentinel_memory.sqlite3") -> None:
        self.path = Path(path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def write_memory(
        self,
        sentinel_input: SentinelInput,
        processed_clause: ProcessedClause,
        embedding: list[float],
    ) -> int:
        matrix = processed_clause.emotional_matrix
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (
                    sender, session_id, source_input_id, source_text, clause_text,
                    emotional_matrix_json, summary, felt_reason, valence, arousal,
                    importance, raw_depth, normalized_depth, embedding_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(sentinel_input.sender),
                    sentinel_input.session_id,
                    sentinel_input.input_id,
                    sentinel_input.text,
                    processed_clause.clause.text,
                    json.dumps(matrix.to_json_dict()),
                    matrix.summary,
                    matrix.felt_reason,
                    matrix.valence,
                    matrix.arousal,
                    matrix.importance,
                    processed_clause.depth.raw,
                    processed_clause.depth.normalized,
                    json.dumps(embedding),
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def list_memories(self) -> list[Memory]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM memories").fetchall()
        return [self._row_to_memory(row) for row in rows]

    def mark_accessed(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE memories
                SET last_accessed_at = ?, access_count = access_count + 1
                WHERE id IN ({placeholders})
                """,
                [now, *memory_ids],
            )

    def add_training_example(
        self, input_text: str, target_matrix: dict[str, Any], source: str
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO training_examples (
                    input_text, target_matrix_json, source, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (input_text, json.dumps(target_matrix), source, now),
            )
            return int(cursor.lastrowid)

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=int(row["id"]),
            sender=row["sender"],
            source_text=row["source_text"],
            clause_text=row["clause_text"],
            emotional_matrix=json.loads(row["emotional_matrix_json"]),
            summary=row["summary"] or "",
            felt_reason=row["felt_reason"] or "",
            valence=float(row["valence"]),
            arousal=float(row["arousal"]),
            importance=float(row["importance"]),
            raw_depth=float(row["raw_depth"]),
            normalized_depth=float(row["normalized_depth"]),
            embedding=json.loads(row["embedding_json"]),
            created_at=row["created_at"],
            access_count=int(row["access_count"] or 0),
        )
