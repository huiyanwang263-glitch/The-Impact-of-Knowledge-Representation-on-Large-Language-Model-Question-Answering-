# D1 2WikiMultihopQA Gold-only representation experiment

This pipeline evaluates 2WikiMultihopQA gold evidence under multiple
serializations:

- `document`: original gold supporting sentences from Wikipedia context
- `triple`: dataset-provided evidence triples
- `graph`: directed graph edge serialization of the same triples

No distractor paragraphs, answer candidates, or retrieval results are shown to
the model.

Important: the current `document` condition is a raw supporting-sentence
condition. It is not always semantically identical to the structured evidence
triples, because 2Wiki supporting sentences can contain extra information or
different answer granularity. For the closest match to the KQA Pro canonical
statement setup, treat 2Wiki `evidences` as the canonical fact layer and add a
separate `document_from_triple` condition before drawing final representation
conclusions.

## Build only

```powershell
python processing\twowiki_gold_only_processing\run_d1_gold_only.py `
  --build-only `
  --per-type 5
```

`--per-type 5` gives 10 questions total. With all three representations, this
creates 30 model calls.

## Run a 10-question pilot for each representation

```powershell
python processing\twowiki_gold_only_processing\run_d1_gold_only.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k `
  --per-type 5 `
  --evidence-order shuffled `
  --max-tokens 4096 `
  --no-think `
  --output-dir processing\twowiki_gold_only_processing\output\d1_gold_only_10q
```

For paired comparison with an existing D0 run, use the same question file:

```powershell
python processing\twowiki_gold_only_processing\run_d1_gold_only.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k `
  --questions-from processing\twowiki_closed_book_processing\output\d0_closed_book_4096\d0_questions.jsonl `
  --evidence-order shuffled `
  --max-tokens 4096 `
  --no-think `
  --output-dir processing\twowiki_gold_only_processing\output\d1_gold_only_paired_d0_40q
```

Keep `--ollama-think-false` off for the current qwen3-4b-16k setup because it
produced empty outputs in D0 smoke tests.

Use `--evidence-order shuffled` when you want the triple/graph conditions to
match the KQA Pro unordered-triple control more closely. The default `dataset`
order preserves 2Wiki's original evidence order, which often follows the
reasoning chain and is therefore less comparable to the KQA Pro shuffled triple
setup.
