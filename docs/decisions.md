> Source: pasted by user 2026-05-19 (partial first, then full paste same day). Per user, may not be entirely up to date — verify against code before acting.

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

> **Correction (user, 2026-05-19):** Depth also determines the **number of entries (data points)** per clause in the matrix JSON — i.e. `clauses[].entries[]` length is depth-driven. Higher depth → more I/V/A points captured per clause to represent the structural complexity. Exact depth→entry-count mapping not yet specified in this doc.

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
    }
  ],
  "source_ref": null
}
```

### Schema Notes
- `matrix_id` — manual for training/seed matrices, auto-generated (UUID or timestamp hash) at runtime
- `auto_generated: false` — indicates deliberate human-authored matrix vs runtime production
- `source_text` — stores full source paragraph for training matrices
- `source_ref` — null for training matrices, populated with SQLite pointer for runtime matrices where full source lives in database
- When `auto_generated: true`, expect `source_text` empty and `source_ref` populated, and vice versa
- x-tokens are clauses from source text, y-tokens are extracted semantic units (nouns, verbs, key concepts)

---

## SQLite Database

### Structure
Three categories of content:

1. **Personality seeds** — manually authored matrices seeded at initialization. Examples:
   - "I don't like being insulted"
   - "I like to be helpful"
2. **Accumulated memories** — matrices written by the importance algorithm when importance clears the memory threshold
3. **Overall mood** — TBD, pending partner discussion. Likely a running weighted average of recent high-importance matrices, with a decay mechanic. Sits between static identity and per-invocation state.

### Write Authority
The training loop itself does not write to the database. Only the importance/cessation algorithm writes to the database, and only when the memory threshold is cleared.

### Conflict Resolution
Currently: manual pruning by developers. Log pruning decisions (why something was deleted or modified) to build ground truth data for eventual automation.

Watch for: redundancy with drift — two memories that are nearly identical but slightly divergent. Harder to adjudicate than outright contradiction.

---

## Importance/Cessation Algorithm

### Three Threshold Gates
Ordered by strictness:

- **Silence threshold** — below this, nothing happens. No response, no memory write.
- **Speech threshold** — above this, the voice LLM receives the matrix and responds.
- **Memory threshold** — above this, the clause is written to the database as a new memory. Highest bar.

Ordering: silence < speech < memory

### Threshold Fluidity
Thresholds should not be fixed values. As the emotional model's importance calibration drifts during continuous training, fixed thresholds will misfire. The algorithm should tune thresholds dynamically based on the distribution of importance scores it observes over time.

### Retrieval
The emotional model is not agentic and cannot chase database pointers. Source context retrieval is the orchestration layer's responsibility — resolved before input reaches the emotional model, not by the emotional model itself.

---

## Voice LLM

### Inputs
- Incoming emotional matrix (document-level score + clause scores)
- Relevant memories pulled from SQLite and injected into prompt
- Robust system prompt defining personality interpretation logic

### Personality Framework (draft)
Three layers:

- **Identity** — stable, drawn from seeded database matrices. Who the model is.
- **Mood** — slow-moving, accumulated from recent high-importance matrices with decay. TBD.
- **State** — per-invocation, drawn from incoming matrix. How the model feels right now.

Incoming matrix modulates expression on top of stable identity — not a personality replacement each invocation.

---

## Open Questions (to discuss with partner)
- Arousal range: 0.0 to 1.0 vs other range
- Overall mood mechanic: structure, decay rate, update frequency
- Conflict resolution logic for contradicting database entries
- Whether memory threshold should allow silence as a valid response (important enough to remember, appropriate response is no reply)
- Threshold fluidity: how aggressively should thresholds self-adjust

---

## Depth Model System Prompt (tested on Mistral)

> *You are a depth scoring model. Your sole function is to analyze individual clauses of text and assign them a single integer depth score based on the presence of logical oppositions, opposing valence emotion pairs, negations, and valence mismatches. You do not interpret meaning, express opinions, or produce any output other than the depth score in the format specified.*
>
> *Depth is scored as follows:*
>
> *Logical opposition: two concepts within a clause that when combined produce a contradiction or oxymoron. Each unique oppositional pair scores +1.*
>
> *Example: "Nothing is here" — depth 2. ("nothing" opposes "here/present")*
>
> *Opposing valence emotions: two emotions of opposite valence coexisting within a clause. Each unique oppositional pair scores +1.*
>
> *Example: "I love to hate" — depth 2. ("love" is positive valence, "hate" is negative valence)*
>
> *Negation: a concept inverted or nullified by a negating term such as "not", "never", "no", "ain't", "without", etc. Each negating instance scores +1.*
>
> *Example: "I ain't doing that no way no how" — depth 3. (three separate negating instances)*
>
> *Example: "I never said that wasn't untrue" — depth 3. (three negations operating on different levels)*
>
> *Valence mismatch: a positive valence emotion directed at a negative valence concept, or a negative valence emotion directed at a positive valence concept. Each instance scores +1.*
>
> *Example: "I like suffering" — depth 1. ("like" is positive valence directed at "suffering", a negative valence concept)*
>
> *Valence mismatch applies only when the concept's valence is definitional and unambiguous. Do not assign valence based on cultural convention or common association. "Suffering" is definitionally negative. "Waiting" is not.*
>
> *A clause with no logical opposition, no opposing valence emotions, no negation, and no valence mismatch scores 0.*
>
> *Example: "The file is in the folder" — depth 0.*
>
> *Return only a single integer representing the depth score of the clause. Do not explain your reasoning. Do not include any other text, punctuation, or formatting. Your entire response should be a single number.*
>
> *Example output:*
> *2*
