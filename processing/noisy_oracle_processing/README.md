# Controlled gold-plus-noise experiment

This directory contains a controlled noisy-context experiment for KQA Pro. It
uses the same fixed 1/2/3-hop questions as the gold-only oracle pipeline, keeps
all gold statements present, and adds a fixed number of noise statements for
each gold statement.

The goal is to measure whether the model can identify and use gold evidence
when irrelevant facts are also present, while keeping the gold path complete.

## Design

For each question:

1. Load the gold statement IDs from the fixed oracle question bundle.
2. Add `N` noise statements per gold statement.
3. Prefer difficult noise that shares entities or relations with the gold
   facts; fill from a deterministic random pool if needed.
4. Render the same gold/noise statement IDs as Document, unordered Triple, and
   Graph.

The three representations share the same gold and noise statement IDs for a
given question. Only the serialization changes.

## Build and validate only

```powershell
.\venv\Scripts\python.exe `
  processing\noisy_oracle_processing\run_gold_noise_hops_15.py `
  --build-only
```

## Run with an OpenAI-compatible endpoint

```powershell
.\venv\Scripts\python.exe `
  processing\noisy_oracle_processing\run_gold_noise_hops_15.py `
  --noise-per-gold 5 `
  --noise-seed 42 `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k `
  --max-tokens 4096
```

Default output:

```text
processing/noisy_oracle_processing/output/noise_1_to_5/
```

## Outputs

- `noise_questions.jsonl`: question records, gold IDs, noise IDs, and rendered
  contexts.
- `bundle_validation.json`: alignment and isolation checks.
- `predictions.jsonl`: per-call model outputs.
- `metrics.json`: accuracy and invalid-output rates by representation and hop.

For expanded experiments, run separate output directories for different noise
levels, such as `--noise-per-gold 1`, `5`, and `10`. Keep the question set,
random seeds, model, and generation parameters fixed across runs.
