# KQA Pro gold-only knowledge representation experiments

This directory contains the KQA Pro canonical-statement generation and
gold-only oracle experiments. The key control is that Document, Triple, and
Graph are rendered from the same underlying canonical statements, so only the
serialization changes.

## Canonical pilot representation builder

`build_pilot_representations.py` builds three aligned representations from a
small KB subgraph:

1. `documents`: natural-language fact documents grouped by subject.
2. `unordered_triples`: the same statements in deterministic shuffled order.
3. `readable_graph`: the same statements as labelled nodes and edges.

Run:

```powershell
.\venv\Scripts\python.exe `
  processing\knowledge_representation_processing\build_pilot_representations.py
```

Default output:

```text
processing/knowledge_representation_processing/output/pilot/
```

Important files:

- `canonical_statements.jsonl`: the single source of truth.
- `documents.jsonl` and `documents.md`: document serialization.
- `unordered_triples.jsonl` and `unordered_triples.txt`: shuffled triples.
- `readable_graph.json` and `readable_graph.md`: graph serialization.
- `manifest.json`: parameters, counts, and alignment checksum.

The generator verifies that all three representations contain exactly the same
statement IDs. Generation fails if the ID sets differ.

## E1 oracle experiment on a fixed 1/2/3-hop pilot

`run_oracle_hops_15.py` selects a fixed set of traceable relation-chain
questions:

```text
1-hop: 5 questions
2-hop: 5 questions
3-hop: 5 questions
```

Each question is run under three complete-gold conditions:

```text
Document / unordered Triple / Graph
```

No answer choices are shown to the model.

Build and validate the bundle only:

```powershell
.\venv\Scripts\python.exe `
  processing\knowledge_representation_processing\run_oracle_hops_15.py `
  --build-only
```

Run with an OpenAI-compatible endpoint:

```powershell
.\venv\Scripts\python.exe `
  processing\knowledge_representation_processing\run_oracle_hops_15.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k `
  --max-tokens 4096
```

Default output:

```text
processing/knowledge_representation_processing/output/oracle_hops_15/
```

Outputs include:

- `oracle_questions.jsonl`: questions, gold statement IDs, and three oracle
  contexts.
- `bundle_validation.json`: alignment and completeness checks.
- `predictions.jsonl`: per-call model outputs and token usage.
- `metrics.json`: overall, per-representation, and per-hop metrics.

## Legacy 20-question oracle smoke test

`run_oracle_20.py` is retained as an older smoke-test script. It samples 20
two-hop questions and runs the same three representation conditions. New
experiments should prefer `run_oracle_hops_15.py`, because it is aligned with
the E0/E1/E2/E3 staged design and reports hop-stratified metrics.
