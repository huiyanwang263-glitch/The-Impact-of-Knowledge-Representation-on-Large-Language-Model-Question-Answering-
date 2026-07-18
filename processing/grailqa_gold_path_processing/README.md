# GrailQA 2-hop/3-hop gold-path materialization

This folder builds grounded Freebase evidence for GrailQA questions before any
manual-noise experiment. It does **not** perform full-KB retrieval.

## Scope

- source: GrailQA dev JSON by default;
- hops: `num_edge` 2 and 3;
- function: `none` only;
- answers: entity answers only;
- acceptance: all graph nodes must be bound and the materialized question-node
  denotation must exactly match the annotated GrailQA answer set.

Here “2-hop/3-hop” follows GrailQA's `num_edge` field. A three-edge query can be
a branched conjunctive graph rather than a simple three-edge chain. The emitted
gold statements contain exactly the instantiated `graph_query.edges`; Freebase
type constraints remain in SPARQL for validation but are not added as extra
evidence statements.

The source `graph_query` is a query graph, not a grounded evidence graph. The
builder rewrites the provided gold SPARQL so its inner `SELECT DISTINCT` returns
all `xN` variables, executes that query, and instantiates every graph edge from
the returned bindings.

## Endpoint

A Freebase SPARQL endpoint aligned with GrailQA is required. Pass it explicitly:

```powershell
python processing/grailqa_gold_path_processing/build_gold_paths.py `
  --endpoint http://127.0.0.1:8890/sparql
```

Alternatively set `GRAILQA_SPARQL_ENDPOINT`. If neither is supplied, the script
uses the legacy endpoint found in the bundled GrailQA baseline code; that
service should not be assumed to remain available.

## Dry run

This validates selection and query rewriting without network access:

```powershell
python processing/grailqa_gold_path_processing/build_gold_paths.py --dry-run
```

Use `--max-per-hop 2` for a small endpoint smoke test. Successful queries are
checkpointed immediately in `query_results.jsonl`, so an interrupted run can be
resumed safely by running the same command again. The run stops after three
consecutive endpoint failures by default, preventing an unavailable service
from producing thousands of repeated failure requests.

## Outputs

The default output directory is `output/dev_hops_2_3/` under this folder.

- `materialization_queries.jsonl`: rewritten all-variable SPARQL queries;
- `query_results.jsonl`: append-only endpoint checkpoints;
- `gold_questions.jsonl`: validated question bundles with grounded facts;
- `canonical_gold_statements.jsonl`: deduplicated global statement bank;
- `failures.jsonl`: rewrite, endpoint, empty-binding, or denotation failures;
- `label_failures.jsonl`: non-fatal friendly-name lookup failures;
- `summary.json`: final coverage and audit counts.

Each statement has a stable ID derived from `(subject_id, relation_id,
object_id)`. A later manual-noise builder must choose gold/noise statement IDs
first and only then serialize the same IDs as Document, Triple, and Graph.
