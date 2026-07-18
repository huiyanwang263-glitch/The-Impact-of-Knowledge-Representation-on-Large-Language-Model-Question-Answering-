# GrailQA E3: question-derived noise with LLM evidence selection

This folder adapts the KQA Pro manual E3 design to the same 30 GrailQA
questions used by E0 and E1. It does not download, index, or retrieve over the
full Freebase dump.

## Noise construction inherited from KQA Pro

The KQA Pro manual E3 builder uses true statements traced from other questions'
Gold paths rather than fabricated false triples. It excludes evaluation
questions from the background pool, removes overlap with evaluation Gold, and
filters noise facts that directly contain the current answer. Candidate IDs and
their order are then shared across Document, Triple, and Graph. Gold is present
in the candidate bank but is not identified or forced into the selector output.

This GrailQA implementation follows the same controls:

1. Read non-evaluation GrailQA train/dev questions.
2. Keep questions whose query graph can be grounded offline from the annotated
   answer MIDs: the only class variable must be the question/answer node.
3. Instantiate their query edges into true Freebase statements.
4. Exclude all 30 evaluation QIDs and every statement that overlaps evaluation
   Gold.
5. For each evaluation question, also exclude noise whose subject or object is
   one of that question's Gold answer MIDs.
6. Deterministically sample exactly 100 noise statements per question.
7. Add the complete sufficient Gold subgraph, shuffle once, and reuse exactly
   the same candidate IDs and order across all representations.
8. Ask the LLM selector for up to 50 statement IDs. E3 never repairs the result
   with missing Gold statements.

No distractor-choice filter is needed because GrailQA is open-ended and has no
multiple-choice distractors.

## Why GrailQA uses 100 noise + Top-50

KQA Pro has only a few Gold facts per question, so its pilot can use a fixed
100-statement bank and Top-20. GrailQA includes multi-answer questions. In this
30-question bundle, one question has 43 correct answers and needs a minimal
sufficient Gold subgraph of 44 statements. Top-20 could never retrieve enough
evidence for that question.

Therefore this implementation fixes the actual noise load at 100 statements
and uses a global Top-50 for every question and representation. Candidate-bank
size varies only with sufficient-Gold size. Both values are recorded in the
validation output.

For queries with multiple equivalent intermediate bindings, E3 does not label
the union of every binding as required Gold. It deterministically keeps one
grounded path per Gold answer and takes their deduplicated union. This is the
smallest audited evidence target that still supports the complete answer set.

## Build and validate only

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e3_manual_llm_retrieval_processing\run_e3_manual_retrieval.py `
  --build-only
```

## Selector-only run

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e3_manual_llm_retrieval_processing\run_e3_manual_retrieval.py `
  --selection-only
```

## Three-question smoke test

This uses the same first question per hop as the E0/E1 smoke tests and runs all
three representations:

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e3_manual_llm_retrieval_processing\run_e3_manual_retrieval.py `
  --max-per-hop 1 `
  --output-dir processing\grailqa_e3_manual_llm_retrieval_processing\output\e3_smoke_1_per_hop
```

## Complete E3 run

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_e3_manual_llm_retrieval_processing\run_e3_manual_retrieval.py
```

Default outputs are written under `output/e3_manual_30/`:

- `question_derived_noise_bank.jsonl`: the deduplicated union of noise facts
  actually sampled into the 30 banks, with source-question provenance. The
  much larger deterministic background-pool count is retained in validation
  but the unused pool is not persisted;
- `candidate_banks.jsonl`: aligned per-question candidate IDs and statements;
- `bundle_validation.json`: leakage, Gold completeness, size, and alignment
  checks;
- `selection_results.jsonl`: selector outputs and retrieval metrics;
- `predictions.jsonl`: E3 answers with canonical and alias-aware scores;
- `e3_checkpoint.jsonl`: append-only resumable checkpoints;
- `metrics.json`: retrieval and answer metrics by representation and hop.
