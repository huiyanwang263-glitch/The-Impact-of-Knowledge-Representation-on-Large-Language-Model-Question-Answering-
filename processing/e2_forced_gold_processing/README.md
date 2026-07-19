# E2 Forced-Gold with E3-M manual noise

This directory contains the KQA Pro E2 experiment paired with the manual-noise
E3-M pipeline:

```text
E1: question + complete gold evidence only
E2: question + saved E3-M selection + forced-complete Gold path
E3: question + saved E3-M selected context only
```

The purpose is to separate missing-evidence errors from answer-reasoning errors
while preserving the exact E3-M candidate bank and raw selector output.

## Context construction

The script reads E3-M `candidate_banks.jsonl`, `selection_results.jsonl`, and
`configuration.json`. It never reruns the selector.

For each question and representation:

1. Validate the E3-M candidate bundle hash and exact candidate order.
2. Reuse the saved E3-M selected statement IDs for each representation.
3. Replace tail non-Gold statements with missing Gold statements.
4. If the raw selection is shorter than the Gold path, expand only to the Gold
   path size; the final context remains within the E3 Top-K.
5. If the selector was invalid, use Gold-only as an audited fallback.
6. Render the repaired IDs using the same renderer and answer prompt as E3-M.
7. Report retrieval metrics before forcing separately from the repaired-context
   metrics.

Graph uses the compact inline edge format:

```text
EDGE Q1 (Entity A) --[relation]--> Q2 (Entity B)
```

The standalone node table is not shown in the prompt, because it can consume
context budget and push key edges out of view.

## Default inputs

- Fixed question bundle:
  `processing/knowledge_representation_processing/output/oracle_hops_15/oracle_questions.jsonl`
- E3-M candidate banks and selections:
  `processing/e3_manual_llm_retrieval_processing/output/e3_manual_hops_15/`

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
  --answer-max-tokens 4096
```

Default output:

```text
processing/e2_forced_gold_processing/output/e2_manual_hops_15/
```

## Outputs

- `e2_contexts.jsonl`: raw E3 selections, forcing records, repaired contexts,
  and before/after retrieval metrics.
- `configuration.json`: E3 candidate and selection hashes plus answer controls.
- `bundle_validation.json`: E3 alignment, Gold completeness, and Top-K checks.
- `predictions.jsonl`: model outputs for full runs.
- `e2_results_checkpoint.jsonl`: resumable per-case answer checkpoints.
- `metrics.json`: retrieval-before-forcing, repaired-context, answer, token, and
  latency metrics by representation and hop.

Each context record keeps the raw E3-M selected IDs, missing Gold, removed
noise, appended Gold, final IDs, and whether the context length changed.

## Interpretation notes

E2 uses the exact statement selection produced by E3-M for each representation.
Therefore:

- E2 versus E3 within the same representation is a paired diagnostic of missing
  Gold evidence.
- Cross-representation E2 comparisons still include differences in the noise
  statements selected by E3-M.
- When a raw selection is shorter than the Gold path, E2 must expand it. The
  recorded context-size delta must be considered when interpreting E2/E3.

The previous Full/Atomic-RAG E2 implementation remains available through Git
history; this script now defines the main manual-noise E2 used with E3-M.
