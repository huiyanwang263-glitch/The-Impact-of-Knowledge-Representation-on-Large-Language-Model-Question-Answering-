# KQA Pro full-KB corpus construction and native RAG

This directory contains utilities for building full-KB corpora from `kb.json`
and running an early native RAG pipeline. The newer atomic RAG pipeline is
documented in [README_atomic.md](README_atomic.md) and is the main controlled
E3 condition.

## Full-KB corpus builder

`build_full_kb_corpora.py` converts the complete KQA Pro KB into aligned
representations:

```text
full KB
  -> canonical statements
     -> Document chunks
     -> globally shuffled Triple chunks
     -> Graph nodes / edges / adjacency chunks
```

Run:

```powershell
.\venv\Scripts\python.exe `
  processing\rag_full_kb_processing\build_full_kb_corpora.py
```

Default output:

```text
processing/rag_full_kb_processing/output/full_kb/
```

Important files:

| File | Purpose |
| --- | --- |
| `manifest.json` | Corpus scale, parameters, and alignment checks |
| `canonical/statements.jsonl` | Deduplicated canonical statement layer |
| `document/documents.jsonl` | Natural-language document chunks |
| `triple/unordered_triples.jsonl` | Deterministically shuffled triple chunks |
| `graph/nodes.jsonl` | Graph node table |
| `graph/edges.jsonl` | Graph edge table, one edge per statement ID |
| `graph/adjacency_chunks.jsonl` | Readable adjacency-list chunks |

The full-KB builder is useful for inspecting native chunking behavior. In the
main controlled experiment, prefer the atomic corpus so that each retrieval
chunk contains exactly one statement.

## Native RAG smoke test

`run_rag_hops_15.py` reads the fixed 15-question oracle bundle and runs BM25
retrieval over the native full-KB corpora.

Retrieval-only run:

```powershell
.\venv\Scripts\python.exe `
  processing\rag_full_kb_processing\run_rag_hops_15.py `
  --retrieval-only
```

Full RAG run:

```powershell
.\venv\Scripts\python.exe `
  processing\rag_full_kb_processing\run_rag_hops_15.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k `
  --max-tokens 4096 `
  --context-word-budget 2000
```

Default output:

```text
processing/rag_full_kb_processing/output/rag_hops_15/
```

Outputs include:

- `retrieval_results.jsonl`: retrieved chunks, scores, statement IDs, and
  recall metrics.
- `predictions.jsonl`: model answers for full RAG runs.
- `metrics.json`: retrieval and answer metrics by hop and representation.
- `configuration.json`: retrieval and generation parameters.

Gold statement IDs are used only after retrieval for offline evaluation. They
must not participate in indexing, scoring, reranking, or context construction.
