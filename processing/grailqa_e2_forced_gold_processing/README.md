# GrailQA E2: noisy retrieval with forced Gold completion

E2 uses exactly the same 30 questions, sufficient-Gold construction, true
question-derived noise pool, per-question sampling, candidate order, saved E3
selector output, and answer prompt as GrailQA E3. The only experimental change
is that E2 repairs the saved E3 selection before answering so that every
sufficient Gold statement is present in the final context.

## Forced-completion rule

1. Reuse the matching raw selector output saved by E3 for the same
   `100 noise + sufficient Gold` candidate bank. The candidate-bundle hash must
   match before E2 will run.
2. Record the raw selector output and its retrieval metrics unchanged.
3. Retain all selected Gold and as many selected non-Gold statements as fit.
4. Append every missing sufficient Gold statement in deterministic Gold order.
5. Keep the original selected-context size whenever it is already at least the
   Gold size. If the selector chose fewer statements than the Gold size, expand
   only to the Gold size. The repaired context therefore never exceeds Top-50.
6. If the selector output is invalid or empty, record that failure and use the
   sufficient Gold set as the answer context. This keeps the E2 Gold guarantee
   without misreporting the failed selector as successful retrieval.

This makes the comparison auditable: `selector_before_forcing` measures actual
retrieval, while `forced_context` describes what the answer model received. E2
must not be interpreted as having perfect retrieval; it has guaranteed Gold
availability after an oracle repair.

Reusing E3 selection is important even at temperature 0 because repeated local
model calls are not guaranteed to produce identical ID sets. The optional
`--run-selector` flag performs a fresh selection only for diagnostics; it is not
the preferred controlled E2-versus-E3 comparison.

## Build and validate only

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e2_forced_gold_processing\run_e2_forced_gold.py `
  --build-only
```

## Three-question smoke test

This selects the same first question from each of the 1-hop, 2-hop, and 3-hop
groups as E0, E1, and E3, and runs Document, Triple, and Graph:

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e2_forced_gold_processing\run_e2_forced_gold.py `
  --max-per-hop 1 `
  --e3-selection-results processing\grailqa_e3_manual_llm_retrieval_processing\output\e3_smoke_1_per_hop\selection_results.jsonl `
  --output-dir processing\grailqa_e2_forced_gold_processing\output\e2_smoke_1_per_hop
```

## Complete 30-question run

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e2_forced_gold_processing\run_e2_forced_gold.py
```

Outputs include the sampled noise union, aligned candidate banks, validation,
raw and repaired selections, predictions, resumable checkpoints, and aggregate
metrics. Generated outputs are ignored by Git.
