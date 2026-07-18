# GrailQA 1/2/3-hop mini knowledge base

This folder builds a small, auditable Freebase knowledge base for a GrailQA
pilot. It does not download, index, or retrieve over the complete Freebase
dump.

The default run materializes exactly ten successful non-aggregate,
entity-answer questions for each of `num_edge = 1, 2, 3`. If a candidate
cannot be executed or validated, the builder continues with another question
at the same hop until the target is reached or the candidate pool is exhausted.

## Data flow

1. Read the local GrailQA dev split.
2. Deterministically shuffle eligible questions with seed 42.
3. Rewrite each provided Gold SPARQL query to return all graph variables.
4. Execute the query against a GrailQA-compatible Freebase endpoint.
5. Require the returned answer denotation to equal the annotated answer set.
6. Instantiate every Gold query-graph edge and deduplicate those statements
   into one mini knowledge base.

The hop number follows GrailQA's `num_edge`. A three-edge query may be a
branched conjunctive graph rather than a simple chain.

## Dry run

This checks local selection and query rewriting without network access:

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_mini_kb_processing\build_mini_kb.py `
  --dry-run
```

## Materialize the mini-KB

Use an endpoint whose Freebase snapshot and conventions are compatible with
GrailQA:

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_mini_kb_processing\build_mini_kb.py `
  --endpoint http://127.0.0.1:8890/sparql
```

A currently available hosted alternative is QLever's read-only Freebase API:

```powershell
C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe `
  processing\grailqa_mini_kb_processing\build_mini_kb.py `
  --endpoint https://qlever.dev/api/freebase
```

If `--endpoint` is omitted, the script tries the legacy endpoint referenced by
the bundled GrailQA tooling. That service is external and may no longer be
available.

## Outputs

The default output directory is `output/dev_10_per_hop/` under this folder.

- `candidate_manifest.jsonl`: deterministic candidate order for each hop;
- `materialization_queries.jsonl`: all rewritten queries attempted or prepared;
- `query_results.jsonl`: append-only success/failure checkpoints;
- `selected_questions.jsonl`: exactly the successfully grounded questions;
- `mini_kb.jsonl`: deduplicated facts with Document, Triple, and Graph text,
  plus source-question provenance;
- `failures.jsonl`: latest failed attempt for each question;
- `summary.json`: selection, coverage, validation, and KB-size audit.

The three representations share the same `statement_id`. This mini-KB can
therefore be used later to construct aligned E1 Gold contexts and fixed-size
E2/E3 candidate banks without full-Freebase retrieval.
