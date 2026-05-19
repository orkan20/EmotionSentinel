# EmotionSentinel

A multi-model pipeline that processes text through emotional analysis, maintains a persistent emotional memory, and (eventually) routes a response through a Voice LLM only when emotional significance clears a threshold. Clauses and the full source text are scored on three axes — **Importance / Valence / Arousal (I/V/A)** — with a depth signal feeding the importance score.

## Target architecture

Five components, in pipeline order (from `docs/decisions.md`):

1. **Depth Model** — per-clause integer depth via four scoring rules (logical opposition, opposing valence, negation, valence mismatch).
2. **Emotional Model** — continuously training; scores each clause and the document on I/V/A.
3. **Importance / Cessation Algorithm** — gates writes and halts training when returns diminish or a user input arrives.
4. **SQLite memory** — personality seeds + accumulated emotional matrices.
5. **Voice LLM** — receives matrix + injected memories, produces the final output.

## Current state (honest)

The pipeline at `sentinel/pipeline.py` runs end-to-end on a smoke test, but most components are placeholders:

- `DepthModel` is **word-count based**, not the 4-rule scorer from the spec.
- `EmotionalModel` is **keyword matching**, not a real model. `test.py` shows the intended replacement: prompting `lzw1008/Emollama-chat-7b` for I/V/A JSON.
- `ThresholdEvaluator` (`sentinel/thresholds.py`), `MatrixBuilder` (`sentinel/matrix_builder.py`), and `OllamaClient` (`sentinel/local_llm.py`) exist but are **not wired into `pipeline.py`**.
- `memory_store.py` + `retriever.py` are implemented, but `DocumentMatrix.memories` is always empty — nothing writes or retrieves yet.
- The **Voice LLM stage does not exist**.

## Quickstart

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python test_pipeline.py
```

Expected output:

```
DocumentMatrix
Has clauses: <n>
Clause type: Clause
```

## Spec

See [`docs/decisions.md`](docs/decisions.md) for the full target spec — depth-scoring rules, schema, and architecture decisions.

## Layout

```
sentinel/             package (pipeline + models + scorers + storage)
Code/Sentinel.py      exploratory clause-extraction sketch
example matrixes/     sample matrix JSON (note: schema predates docs/decisions.md)
test.py               Emollama-chat-7b standalone smoke test
test_pipeline.py      SentinelPipeline smoke test
docs/decisions.md     target architecture spec
```
