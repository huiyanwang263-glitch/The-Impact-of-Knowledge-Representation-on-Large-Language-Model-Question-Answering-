# GrailQA E0 closed-book baseline

This folder runs the E0 condition on the grounded questions selected by
`grailqa_mini_kb_processing`. The model receives only the natural-language
question. No mini-KB facts, Gold paths, SPARQL, logical forms, entity IDs,
retrieval results, or answer candidates are included in the prompt.

GrailQA questions can have more than one correct answer. The model therefore
returns a JSON answer array. The script reports two parallel score groups:

- `canonical`: strict normalized matching against GrailQA's canonical answer
  names;
- `alias_aware`: one-to-one matching from predicted surfaces to Gold Freebase
  MIDs using a frozen alias cache.

Both groups contain answer-set exact match, precision, recall, and F1. Aliases
are evaluation-only and are never added to the model prompt.

## Build the answer alias cache

Run this once before E0/E1/E2/E3 evaluation:

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e0_processing\build_answer_alias_cache.py
```

The builder only collects aliases for Gold MIDs in the frozen 30-question
bundle. It uses canonical answer names and graph-node names from local GrailQA
train/dev, plus repeated mentions and friendly names from the bundled entity
linking results. One-off entity-linker mentions are excluded by default. Every
accepted alias keeps its source and count, and the cache records a SHA-256 hash.

## Three-question smoke test

Run one question from each hop:

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e0_processing\run_e0_closed_book.py `
  --base-url http://127.0.0.1:11434 `
  --model qwen3-4b-16k `
  --max-per-hop 1 `
  --output-dir processing\grailqa_e0_processing\output\e0_smoke_1_per_hop
```

## Full 30-question run

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e0_processing\run_e0_closed_book.py `
  --base-url http://127.0.0.1:11434 `
  --model qwen3-4b-16k
```

Default outputs are written under `output/e0_30/`:

- `e0_questions.jsonl`: the frozen evaluation bundle without evidence;
- `bundle_validation.json`: checks that the E0 prompt excludes knowledge;
- `predictions.jsonl`: raw structured responses and per-question scores;
- `metrics.json`: aggregate metrics overall and by hop.

The shared alias cache is written to
`processing/grailqa_e0_processing/output/answer_alias_cache.json`.
