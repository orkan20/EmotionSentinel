# EmotionSentinel

A multi-model pipeline that processes text through emotional analysis, maintains a persistent emotional memory, and (eventually) routes a response through a Voice LLM only when emotional significance clears a threshold. Clauses and the full source text are scored on three axes — **Importance / Valence / Arousal (I/V/A)** — with a depth signal feeding the importance score.

## Target architecture

Five components, in pipeline order (from `docs/decisions.md`):

1. **Depth Model** — per-clause integer depth via four scoring rules (logical opposition, opposing valence, negation, valence mismatch).
2. **Emotional Model** — continuously training; scores each clause and the document on I/V/A.
3. **Importance / Cessation Algorithm** — gates writes and halts training when returns diminish or a user input arrives.
4. **SQLite memory** — personality seeds + accumulated emotional matrices.
5. **Voice LLM** — receives matrix + injected memories, produces the final output.

## Current state

The pipeline at `sentinel/pipeline.py` runs end-to-end. Six spec components are wired; the Voice LLM stage and a few smaller pieces are still pending.

### Wired
- **`ThresholdEvaluator`** (`sentinel/thresholds.py`) — three gates (silence < speech < memory) with fluid self-tuning via `observe()` over a rolling window of importance scores. Fires per-clause and per-document. Covered by `test_thresholds.py` (12 tests, pure Python).
- **Document-level matrix scoring** — `pipeline.process()` does a separate holistic `emotional_model.evaluate()` call over the full source text and stores the result as `DocumentMatrix.document_score` + `document_route_action`. Spec calls this "the primary input to the importance threshold gate."
- **Memory write path** — when a clause's `route_action` is `MEMORY` or `SPEECH_AND_MEMORY`, it's persisted to SQLite by the cessation algorithm (only writer to the DB, per spec).
- **Memory retrieval** — top-K relevant memories pulled up front (orchestration layer responsibility per spec, before the emotional model runs) and attached to `DocumentMatrix.memories`. Ranked by similarity × importance × recency via `MemoryRetriever`.

### Opt-in real LLMs (not the pipeline default)
- **`MistralDepthModel`** (`sentinel/depth.py`) — uses the spec's tested-on-Mistral system prompt verbatim, served via local Ollama. Defensive integer parsing handles LLM drift. Opt in with `SentinelPipeline(depth_model=MistralDepthModel())`. Smoke test: `test_depth_mistral.py` (needs Ollama + `ollama pull mistral`).
- **`EmollamaEmotionalModel`** (`sentinel/emotion.py`) — uses `lzw1008/Emollama-chat-7b` with the Llama-2 `[INST]` prompt format. Lazy-loads transformers/torch on first call (~14GB download first time). Opt in with `SentinelPipeline(emotional_model=EmollamaEmotionalModel())`. Smoke test: `test_emotion_emollama.py`.

### Still mock / orphaned
- **`MockDepthModel`** is the pipeline default — word-count, not the 4-rule scorer.
- **`MockEmotionalModel`** is the pipeline default — keyword matching that produces out-of-range I/V/A values (e.g., importance > 1.0). Threshold logic still routes correctly on these but the numbers themselves aren't meaningful until the real model is enabled.
- **`MatrixBuilder`** (`sentinel/matrix_builder.py`) — implemented but not called by `pipeline.py`.
- **Voice LLM stage** — does not exist. Spec calls for a three-layer personality framework (identity / mood / state); none of it is built.
- **`example matrixes/zA.JSON`** — older `Entries` dict shape, not the `document_score` / `tokens.x/y` / `clauses[]` schema in `docs/decisions.md`.
- **Training-halt logic** in the cessation algorithm — only matters once continuous training of the emotional model exists.

## Quickstart

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# End-to-end smoke (uses mock scorers; creates sentinel_memory.sqlite3 in CWD)
python test_pipeline.py

# Threshold logic, no LLM, no deps beyond stdlib + this package
python test_thresholds.py
```

Expected `test_pipeline.py` output (first run, empty DB):

```
DocumentMatrix
Has clauses: 2
Clause type: ProcessedClauseMatrix
Routes: [(0.8, 'speech_and_memory'), (1.6, 'speech_and_memory')]
Document score: (2.67, 0.34, 0.99)
Document route: speech_and_memory
Memories retrieved: 0
```

Second run picks up memories written by the first.

Smoke tests for the opt-in real LLMs need extra infra:

```bash
# Ollama running locally with mistral pulled
python test_depth_mistral.py

# transformers + torch + ~14GB free disk (HF cache)
python test_emotion_emollama.py
```

## Spec

See [`docs/decisions.md`](docs/decisions.md) for the full target spec — depth-scoring rules, matrix JSON schema, threshold model, personality framework, and the tested-on-Mistral depth system prompt.

## Layout

```
sentinel/                      package (pipeline + models + scorers + storage)
Code/Sentinel.py               exploratory clause-extraction sketch
example matrixes/              sample matrix JSON (schema predates docs/decisions.md)
docs/decisions.md              target architecture spec
test.py                        Emollama-chat-7b raw HF smoke test (predates EmollamaEmotionalModel)
test_pipeline.py               end-to-end SentinelPipeline smoke
test_thresholds.py             ThresholdEvaluator unit tests (pure Python)
test_depth_mistral.py          MistralDepthModel smoke (needs Ollama)
test_emotion_emollama.py       EmollamaEmotionalModel smoke (needs HF/torch + ~14GB)
sentinel_memory.sqlite3        runtime DB, git-ignored
```
