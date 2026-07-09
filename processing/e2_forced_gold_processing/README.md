# E2 Forced-Gold with retrieved noise

This directory contains the KQA Pro E2 experiment. E2 sits between E1
Gold-only and E3 Atomic RAG:

```text
E1: question + complete gold evidence only
E2: question + E3 retrieved noise + forced-complete gold path
E3: question + retrieved context only
```

The purpose is to measure whether the model can identify and use the complete
gold path when realistic retrieved noise is still present.

## Context construction

The script reads E3 `retrieval_results.jsonl` and does not rerun BM25.

For each question and representation:

1. Read the E3 top statement IDs in retrieval order.
2. Deduplicate while preserving rank order.
3. Check which gold statement IDs were already retrieved.
4. For each missing gold statement, remove the lowest-ranked non-gold
   statement.
5. Insert the missing gold statement at the removed statement's position.
6. Verify that the final context still contains the same number of statements.
7. Verify that complete-path recall is 100%.
8. Render the final statement IDs as Document, unordered Triple, or Graph.

Graph uses the compact inline edge format:

```text
EDGE Q1 (Entity A) --[relation]--> Q2 (Entity B)
```

The standalone node table is not shown in the prompt, because it can consume
context budget and push key edges out of view.

## Default inputs

- Fixed question bundle:
  `processing/knowledge_representation_processing/output/oracle_hops_15/oracle_questions.jsonl`
- E3 retrieval results:
  `processing/rag_full_kb_processing/output/rag_atomic_hops_15/retrieval_results.jsonl`
- Atomic KB corpus:
  `processing/rag_full_kb_processing/output/atomic_kb/`

## Build and validate only

```powershell
.\venv\Scripts\python.exe `
  processing\e2_forced_gold_processing\run_e2_forced_gold_hops_15.py `
  --build-only
```

## Run E2

```powershell
.\venv\Scripts\python.exe `
  processing\e2_forced_gold_processing\run_e2_forced_gold_hops_15.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k `
  --max-tokens 4096
```

Default output:

```text
processing/e2_forced_gold_processing/output/e2_atomic_hops_15/
```

## Outputs

- `e2_contexts.jsonl`: constructed contexts, replacement records, and final
  statement IDs.
- `bundle_validation.json`: gold completeness and context-size checks.
- `predictions.jsonl`: model outputs for full runs.
- `metrics.json`: answer metrics, token usage, and latency by representation
  and hop.

Each context record keeps the original retrieved statements, missing gold
statements, removed noise statements, replacement positions, and final
rendered context for auditability.

## Interpretation notes

E2 uses native retrieved noise for each representation. Therefore:

- E2 vs E3 within the same representation is the cleanest diagnostic pair.
- Cross-representation comparisons in E2 still include differences in the
  retrieved noise statement sets.

For a stricter "same noise, different serialization" test, add a separate
E2-Shared-Noise condition where all representations share one statement-ID
set and are rendered separately.
