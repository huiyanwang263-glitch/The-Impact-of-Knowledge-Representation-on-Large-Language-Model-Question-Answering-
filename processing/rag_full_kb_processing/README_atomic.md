# E3 Atomic RAG

The atomic RAG pipeline is the main controlled KQA Pro E3 condition. It removes
chunk-granularity confounds by enforcing:

```text
1 canonical statement = 1 retrieval chunk
```

For the same fact, the Document, Triple, and Graph chunks share the same
`chunk_id` and `statement_id`; only the text serialization differs.

Example:

```text
Document: Paris has the relation 'capital of' to France.
Triple:   (Paris | capital of | France)
Graph:    EDGE Q90 (Paris) --[capital of]--> Q142 (France)
```

## 1. Build the atomic corpora

```powershell
.\venv\Scripts\python.exe `
  processing\rag_full_kb_processing\build_atomic_kb_corpora.py
```

Default output:

```text
processing/rag_full_kb_processing/output/atomic_kb/
```

This step does not read questions, gold answers, or retrieval results, and it
does not call the model.

## 2. Run retrieval only

```powershell
.\venv\Scripts\python.exe `
  processing\rag_full_kb_processing\run_rag_atomic_hops_15.py `
  --retrieval-only `
  --top-statements 100
```

## 3. Run full E3 RAG

```powershell
.\venv\Scripts\python.exe `
  processing\rag_full_kb_processing\run_rag_atomic_hops_15.py `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3-4b-16k `
  --max-tokens 4096 `
  --top-statements 100
```

Default output:

```text
processing/rag_full_kb_processing/output/rag_atomic_hops_15/
```

## Retrieval configuration

The three representations use the same BM25 implementation, tokenizer,
parameters, raw question query, and stable tie-breaking. Each representation is
retrieved independently, and the final context contains the top `N` statement
chunks, where `N` defaults to 100.

The experiment fixes the number of facts, not the number of tokens. Prompt
tokens, completion tokens, and latency are measured as dependent variables.

## Metrics

- `statement_recall`: fraction of gold statements in the final top-`N`
  context.
- `complete_path_recall`: whether all gold statements for the question are in
  the final context.
- `retrieval_mrr`: MRR of gold statements within the initial ranked list.
- `context_mrr`: MRR of gold statements within the final top-`N` context.
- `first_gold_rank`: rank of the first retrieved gold statement.
- Answer accuracy, invalid-output rate, token usage, and model latency.

The older native RAG pipeline is retained for comparison with realistic larger
chunks. The atomic pipeline should be used when isolating the effect of
knowledge representation.
