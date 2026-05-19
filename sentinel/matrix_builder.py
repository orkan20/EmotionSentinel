from __future__ import annotations

from sentinel.models import Memory, ProcessedClause, SentinelInput


class MatrixBuilder:
    def build(
        self,
        sentinel_input: SentinelInput,
        clauses: list[ProcessedClause],
        relevant_memories: list[Memory],
    ) -> dict:
        should_speak = any("speech" in processed.action.value for processed in clauses)
        dominant = self._dominant_matrix(clauses)
        return {
            "input": {
                "id": sentinel_input.input_id,
                "sender": str(sentinel_input.sender),
                "text": sentinel_input.text,
                "session_id": sentinel_input.session_id,
                "created_at": sentinel_input.created_at,
            },
            "clauses": [
                {
                    "id": processed.clause.id,
                    "text": processed.clause.text,
                    "position": processed.clause.position,
                    "depth": {
                        "raw": processed.depth.raw,
                        "normalized": processed.depth.normalized,
                    },
                    "emotional_matrix": processed.emotional_matrix.to_json_dict(),
                    "action": processed.action.value,
                }
                for processed in clauses
            ],
            "dominant_emotional_state": dominant,
            "relevant_memories": [
                {
                    "id": memory.id,
                    "summary": memory.summary,
                    "felt_reason": memory.felt_reason,
                    "importance": memory.importance,
                    "created_at": memory.created_at,
                    "matrix": memory.emotional_matrix,
                }
                for memory in relevant_memories
            ],
            "response_directive": {
                "should_speak": should_speak,
                "voice_output_contract": "text_only",
            },
        }

    @staticmethod
    def _dominant_matrix(clauses: list[ProcessedClause]) -> dict:
        if not clauses:
            return {"valence": 0.0, "arousal": 0.0, "importance": 0.0}
        dominant = max(
            clauses, key=lambda processed: processed.emotional_matrix.importance
        )
        return dominant.emotional_matrix.to_json_dict()
