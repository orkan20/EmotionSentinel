> Source: pasted by user 2026-05-19; per user, "not entirely up to date" — verify against code before acting.

# Emotional Model Project — Decisions Log

## Architecture Overview

A multi-model pipeline that processes text through emotional analysis, maintains a persistent memory database, and produces contextually appropriate responses only when emotional significance clears a threshold.

### Components (in pipeline order):
1. **Depth Model** — analyzes individual clauses, outputs a single integer depth score
2. **Emotional Model** — continuously training, scores clauses and full text on I/V/A axes using depth as a variable
3. **Importance/Cessation Algorithm** — manages training halts, threshold gates, and database writes
4. **SQLite Database** — stores personality seeds and accumulated memories as emotional matrices
5. **Voice LLM** — receives emotional matrix + injected memories, produces final output

---

## Depth Model

### Definition
Depth measures semantic tension within a single clause. Four scoring categories, each contributing +1 per instance or pair:

1. **Logical opposition** — two concepts combining to form a contradiction or oxymoron
   - Example: "Nothing is here" — depth 2
2. **Opposing valence emotions** — two emotions of opposite valence coexisting in a clause
   - Example: "I love to hate" — depth 2
3. **Negation** — a concept inverted or nullified by a negating term (not, never, no, ain't, without, etc.)
   - Example: "I ain't doing that no way no how" — depth 3
   - Example: "I never said that wasn't untrue" — depth 3
4. **Valence mismatch** — a positive valence emotion directed at a negative valence concept, or vice versa. Scores +1 per instance (not per pair), reflecting lower structural intensity than full oxymoron.
   - Example: "I like suffering" — depth 1

### Depth 0
A clause with none of the above scores 0.
- Example: "The file is in the folder" — depth 0

### Critical Rule
The depth model must not assign valence based on cultural convention or common association. Valence mismatch applies only when a concept's valence is definitional and unambiguous.
- "Suffering" is definitionally negative — valence mismatch applies
- "Waiting" is conventionally negative but not definitionally — valence mismatch does not apply
- "I love to eat human" — depth 0. "Human" carries no definitional negative valence.

### Output Format
A single integer. No explanation, no punctuation, no additional text.

---

## Emotional Model

### Scoring Axes
Each clause and the full source text are scored on three axes:

- **Importance (I)** — scalar, range 0.0 to 1.0
- **Valence (V)** — bidirectional, range -1.0 to 1.0 (negative = bad, positive = good)
- **Arousal (A)** — scalar, range TBD (pending partner discussion, likely 0.0 to 1.0)

Depth feeds into importance specifically, not valence. A structurally tense clause carries more weight, not a different emotional direction.

> **Correction (user, 2026-05-19):** Depth also determines the **number of entries (data points)** per clause in the matrix JSON — i.e. `clauses[].entries[]` length is depth-driven. Higher depth → more I/V/A points captured per clause to represent the structural complexity. Exact depth→entry-count mapping not yet specified; pending the rest of the truncated decisions log.

### Scoring Granularity
- Each matrix covers a single clause
- A separate document-level matrix scores the full source text holistically (not an average of clause scores)
- The document-level score is the primary input to the importance threshold gate

### Continuous Training
The emotional model trains continuously on emotional matrix JSONs. Training halts when:
- The importance/cessation algorithm determines returns are too diminishing
- The user provides an input to the system

**Catastrophic forgetting risk:** consider LoRA or parameter-efficient fine-tuning to prevent base model degradation during extended training on a narrow emotional corpus.

---

## Matrix JSON Schema

```json
{
  "matrix_id": "",
  "auto_generated": false,
  "source_text": "",

  "document_score": {
    "importance": 0.0,
    "valence": 0.0,
    "arousal": 0.0
  },

  "tokens": {
    "x": {},
    "y": {}
  },

  "clauses": [
    {
      "depth": 0,
      "entries": [
        {
          "id": "",
          "importance": 0.0,
          "valence": 0.0,
          "arousal": 0.0
        }
      ]
```

<!-- TRUNCATED: paste cut off here. ~116 more lines reportedly exist in the source decisions log; user noted the document is "not entirely up to date." Append the remainder when received. -->
