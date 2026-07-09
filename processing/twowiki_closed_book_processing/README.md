# D0 2WikiMultihopQA Closed-book baseline

This pipeline samples 2WikiMultihopQA questions and asks the model to answer
from the question alone. It does not provide Wikipedia paragraphs, supporting
facts, evidence triples, retrieval results, or answer candidates.

Default sampling targets the two question types that are closest to relation
chain reasoning:

```text
compositional
inference
```

Both are filtered to `evidence_count = 2` by default, so the pilot acts as a
clean two-hop closed-book baseline.

## Build sampled question bundle only

```powershell
.\venv\Scripts\python.exe `
  processing\twowiki_closed_book_processing\run_d0_closed_book.py `
  --build-only `
  --per-type 50
```

## Run with an OpenAI-compatible endpoint

```powershell
.\venv\Scripts\python.exe `
  processing\twowiki_closed_book_processing\run_d0_closed_book.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k `
  --per-type 50 `
  --max-tokens 4096 `
  --no-think
```

For Qwen3-style thinking models, `--no-think` is recommended for this
answer-only baseline. Without it, a small `--max-tokens` value can be consumed
entirely by hidden thinking text, leaving an empty final answer. In the current
qwen3-4b-16k OpenAI-compatible setup, `--ollama-think-false` produced empty
outputs during smoke tests, so keep it off unless you have verified that your
serving endpoint handles the flag correctly.

Outputs are written to:

```text
processing/twowiki_closed_book_processing/output/d0_closed_book/
```

The output bundle contains:

- `d0_questions.jsonl`: sampled questions and gold answers for evaluation.
- `bundle_validation.json`: confirms the no-context/no-retrieval condition.
- `predictions.jsonl`: model answers, answer EM/F1, token usage, and latency.
- `metrics.json`: aggregate metrics overall, by type, and by evidence count.
- `official_prediction_format.json`: answer-only predictions in 2Wiki format;
  supporting facts and evidence predictions are empty because D0 does not ask
  the model to produce them.
