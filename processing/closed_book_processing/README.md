# E0 Closed-book baseline

This directory contains the KQA Pro closed-book baseline. It is independent of
the gold-only, noisy-context, and RAG pipelines. The prompt contains only the
question; no KB facts, gold statements, retrieved context, or answer choices are
shown to the model.

The purpose of E0 is to estimate how much of the fixed question set can be
answered from the model's parametric knowledge or language priors alone.

## Build the question bundle only

```powershell
.\venv\Scripts\python.exe `
  processing\closed_book_processing\run_e0_closed_book_hops_15.py `
  --build-only
```

## Run the baseline

```powershell
.\venv\Scripts\python.exe `
  processing\closed_book_processing\run_e0_closed_book_hops_15.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k `
  --max-tokens 4096
```

The default output directory is:

```text
processing/closed_book_processing/output/e0_hops_15/
```

## Outputs

- `closed_book_questions.jsonl`: the fixed questions and evaluation fields.
- `bundle_validation.json`: confirms that no external knowledge, retrieval
  results, gold statements, or choices are included in the prompt.
- `predictions.jsonl`: per-question answers, raw responses, token usage, and
  latency.
- `metrics.json`: overall and per-hop accuracy, unmatched-output rate, token
  usage, and latency.

E0 has no Document, Triple, or Graph condition because it contains no external
knowledge representation. It is the shared closed-book baseline for all
knowledge-representation conditions.
