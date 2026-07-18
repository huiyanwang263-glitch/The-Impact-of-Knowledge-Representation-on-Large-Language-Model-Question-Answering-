# GrailQA E1 Gold-only representation experiment

This folder runs E1 on exactly the same 30 questions used by GrailQA E0. E1
provides the complete grounded Gold fact set and no other facts. There is no
retrieval and no noise.

Each question is evaluated independently under three representations:

- `document`: one natural-language sentence per Gold statement;
- `triple`: one `(subject, relation, object)` tuple per Gold statement;
- `graph`: one labeled directed edge per Gold statement.

The three conditions share the same statement IDs and statement order. The
model returns a JSON answer array, and the script uses the same frozen alias
cache and parallel canonical/alias-aware metrics as E0. The alias cache is
evaluation-only and is never included in a prompt.

## Build and validate the 90 cases

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e1_gold_only_processing\run_e1_gold_only.py `
  --build-only
```

## Three-question smoke test

This runs one question per hop under all three representations, for nine model
calls:

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e1_gold_only_processing\run_e1_gold_only.py `
  --max-per-hop 1 `
  --output-dir processing\grailqa_e1_gold_only_processing\output\e1_smoke_1_per_hop
```

## Full run

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e1_gold_only_processing\run_e1_gold_only.py
```

Default outputs are written under `output/e1_gold_30/`:

- `e1_cases.jsonl`: frozen questions and the three aligned Gold contexts;
- `bundle_validation.json`: Gold-only and cross-representation alignment audit;
- `predictions.jsonl`: per-case raw outputs and both score groups;
- `metrics.json`: overall, representation-level, and hop-level metrics.

