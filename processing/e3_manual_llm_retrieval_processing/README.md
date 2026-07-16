# KQA Pro E3-M: manual candidate bank with LLM evidence selection

This directory adds a bounded, question-derived E3 experiment without changing
the existing Atomic RAG implementation.

Run the commands below from the `final-project` repository root.

## Experimental condition

For each of the fixed 15 Oracle questions (five questions for each of 1/2/3
hops):

1. Keep the complete gold path in a 100-statement candidate bank.
2. Fill the bank with true statements traced from other, non-evaluation KQA Pro
   questions. No generated full-KB RAG corpus is read.
3. Render exactly the same candidate IDs, in exactly the same order, as
   Document, Triple, and Graph.
4. Ask an LLM selector to return one to 20 unique, relevant statement IDs in
   ranked order.
5. Give only those selected statements to the answer model.

Gold is available in the candidate bank, but E3 never identifies it or forces
it into the selected evidence. The selector may return fewer than 20 facts when
it considers only a small subset relevant; the actual selected count is saved
and reported rather than padded with gold-aware or random facts. This makes
statement recall and complete-path recall measurable without artificial repair.

## Shared question requirement

The script requires the existing E1 question bundle:

```text
processing/knowledge_representation_processing/output/
  oracle_hops_15/oracle_questions.jsonl
```

It validates that the file contains exactly 15 questions with five questions
per hop. E3 deliberately does not resample questions. If the bundle is absent,
build E1 first with the existing Oracle script.

## Defaults

```text
evaluation questions:     15
questions per hop:         5
background paths per hop:   50
candidate statements:      100
selector maximum K:        20
selector output tokens:    512
selector API:              native Ollama structured JSON (`think: false`)
noise seed:                42
representations:           document, triple, graph
```

The background noise bank excludes all evaluation `train_index` values. Noise
facts that mention the correct answer or any distractor choice are also
excluded from that question's candidate bank.

## Build candidate banks only

```powershell
py -3 `
  processing\e3_manual_llm_retrieval_processing\run_e3_manual_llm_retrieval_hops_15.py `
  --build-only
```

## Run selector only

This produces retrieval metrics and is sufficient input for a future E2
forced-gold builder:

```powershell
py -3 `
  processing\e3_manual_llm_retrieval_processing\run_e3_manual_llm_retrieval_hops_15.py `
  --selection-only `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k
```

## Run complete E3

```powershell
py -3 `
  processing\e3_manual_llm_retrieval_processing\run_e3_manual_llm_retrieval_hops_15.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k `
  --selector-api ollama-native `
  --selector-max-tokens 512 `
  --answer-max-tokens 4096
```

## Outputs

Default output directory:

```text
processing/e3_manual_llm_retrieval_processing/output/e3_manual_hops_15/
```

- `question_derived_noise_bank.jsonl`: deduplicated true facts from background
  question paths and their source question IDs;
- `candidate_banks.jsonl`: 15 audited candidate banks, shared across all
  representations;
- `selection_results.jsonl`: selector outputs, selected IDs, retrieval metrics,
  format validity, token usage, and latency;
- `e3_results_checkpoint.jsonl`: append-only per-question/per-representation
  checkpoints used to resume an interrupted run with the same configuration;
- `predictions.jsonl`: E3 answers from selected evidence;
- `configuration.json`: experiment controls and candidate-bundle hash;
- `bundle_validation.json`: question counts, gold completeness, disjointness,
  uniqueness, and alignment checks;
- `metrics.json`: retrieval and answer results by representation and hop.

## Interpretation

Compare Document/Triple/Graph retrieval metrics in E3 to measure whether the
representation changes LLM evidence selection. A paired E2 should read this
experiment's `selection_results.jsonl`, remove the lowest-ranked non-gold facts,
and insert missing gold facts while keeping Top-K fixed at 20.
