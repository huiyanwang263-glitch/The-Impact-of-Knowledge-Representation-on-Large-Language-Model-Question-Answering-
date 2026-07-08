#!/usr/bin/env python3
"""Controlled E3 RAG: aligned atomic chunks and fixed Top-N statements."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DATASET = HERE.parent
ORACLE_DIR = DATASET / "knowledge_representation_processing"
sys.path.insert(0, str(ORACLE_DIR))

import run_oracle_20 as oracle  # noqa: E402
import run_rag_hops_15 as native  # noqa: E402


REPS = ("document", "triple", "graph")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def retrieval_metrics(
    ranked: list[dict[str, Any]], selected: list[dict[str, Any]], gold_ids: list[str]
) -> dict[str, Any]:
    gold = set(gold_ids)
    retrieved = {sid for row in selected for sid in row["statement_ids"]}
    covered = gold & retrieved
    first_retrieval_rank = next(
        (rank for rank, row in enumerate(ranked, 1) if gold & set(row["statement_ids"])),
        None,
    )
    first_context_rank = next(
        (rank for rank, row in enumerate(selected, 1) if gold & set(row["statement_ids"])),
        None,
    )
    return {
        "gold_statement_count": len(gold),
        "retrieved_statement_count": len(retrieved),
        "covered_gold_statement_ids": sorted(covered),
        "statement_recall": len(covered) / len(gold),
        "hit": bool(covered),
        "complete_path_recall": gold <= retrieved,
        "first_gold_retrieval_rank": first_retrieval_rank,
        "retrieval_reciprocal_rank": (
            1 / first_retrieval_rank if first_retrieval_rank else 0.0
        ),
        "first_gold_context_rank": first_context_rank,
        "context_reciprocal_rank": 1 / first_context_rank if first_context_rank else 0.0,
    }


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]], model_ran: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "questions": len(rows),
        "mean_statement_recall": average(
            [row["retrieval_metrics"]["statement_recall"] for row in rows]
        ),
        "hit_rate": average([float(row["retrieval_metrics"]["hit"]) for row in rows]),
        "complete_path_recall": average(
            [float(row["retrieval_metrics"]["complete_path_recall"]) for row in rows]
        ),
        "retrieval_mrr": average(
            [row["retrieval_metrics"]["retrieval_reciprocal_rank"] for row in rows]
        ),
        "context_mrr": average(
            [row["retrieval_metrics"]["context_reciprocal_rank"] for row in rows]
        ),
        "mean_context_statements": average(
            [float(row["context_statement_count"]) for row in rows]
        ),
        "mean_context_words": average([float(row["context_word_count"]) for row in rows]),
    }
    if model_ran:
        correct = sum(bool(row["correct"]) for row in rows)
        invalid = sum(row["prediction"] is None for row in rows)
        valid = len(rows) - invalid
        result["answer"] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows) if rows else None,
            "invalid_outputs": invalid,
            "accuracy_on_valid_outputs": correct / valid if valid else None,
        }
        result["efficiency"] = {
            "mean_latency_seconds": average([row["latency_seconds"] for row in rows]),
            "mean_prompt_tokens": average(
                [float(row["usage"].get("prompt_tokens", 0)) for row in rows]
            ),
            "mean_completion_tokens": average(
                [float(row["usage"].get("completion_tokens", 0)) for row in rows]
            ),
            "mean_total_tokens": average(
                [float(row["usage"].get("total_tokens", 0)) for row in rows]
            ),
        }
    return result


def build_report(rows: list[dict[str, Any]], model_ran: bool, config: dict) -> dict:
    hops = sorted({row["hop_count"] for row in rows})
    reps = [rep for rep in REPS if any(row["representation"] == rep for row in rows)]
    return {
        "status": "completed" if model_ran else "retrieval_only",
        "condition": "E3_atomic_aligned_chunks",
        "questions": len({row["question_id"] for row in rows}),
        "retrieval_runs": len(rows),
        "model_calls": len(rows) if model_ran else 0,
        "configuration": config,
        "by_representation": {
            rep: summarize([row for row in rows if row["representation"] == rep], model_ran)
            for rep in reps
        },
        "by_hop": {
            str(hop): {
                rep: summarize(
                    [
                        row
                        for row in rows
                        if row["hop_count"] == hop and row["representation"] == rep
                    ],
                    model_ran,
                )
                for rep in reps
            }
            for hop in hops
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ORACLE_DIR / "output" / "oracle_hops_15" / "oracle_questions.jsonl",
    )
    parser.add_argument("--corpus-root", type=Path, default=HERE / "output" / "atomic_kb")
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "output" / "rag_atomic_hops_15"
    )
    parser.add_argument("--top-statements", type=int, default=100)
    parser.add_argument("--max-ranked-statements", type=int, default=1000)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--representations", nargs="+", choices=REPS, default=list(REPS))
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = list(read_jsonl(args.questions))
    if not questions:
        raise RuntimeError(f"No questions: {args.questions}")
    vocabulary = {
        token for item in questions for token in native.tokens(item["question"])
    }
    model_ran = not args.retrieval_only and bool(args.base_url and args.model)
    config = {
        "retriever": "BM25",
        "chunk_unit": "one_canonical_statement",
        "same_chunk_ids_for_all_representations": True,
        "selection": f"top_{args.top_statements}_statements",
        "same_statement_count_limit_for_all_representations": True,
        "token_budget_equalized": False,
        "tokens_and_latency_are_outcomes": True,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "query": "original question only",
        "gold_used_during_retrieval": False,
        "graph_serialization": (
            "EDGE source_id (source_label) --[predicate]--> target_id (target_label)"
        ),
        "model": args.model if model_ran else None,
    }

    rows: list[dict[str, Any]] = []
    for rep in args.representations:
        corpus = args.corpus_root / rep / "atomic_statements.jsonl"
        print(f"Building atomic BM25 for {rep}...", flush=True)
        retriever = native.BM25(corpus, rep, vocabulary, args.bm25_k1, args.bm25_b)
        for number, item in enumerate(questions, 1):
            ranked = retriever.search(item["question"], args.max_ranked_statements)
            selected = ranked[: args.top_statements]
            context = "\n".join(row["text"] for row in selected)
            retrieval = retrieval_metrics(ranked, selected, item["gold_statement_ids"])
            row: dict[str, Any] = {
                "question_id": item["question_id"],
                "hop_count": item["hop_count"],
                "representation": rep,
                "question": item["question"],
                "gold_answer": item["answer"],
                "gold_statement_ids": item["gold_statement_ids"],
                "selected_chunks": [
                    {
                        "rank": rank,
                        "chunk_id": chunk["chunk_id"],
                        "score": chunk["score"],
                        "statement_ids": chunk["statement_ids"],
                        "text": chunk["text"],
                    }
                    for rank, chunk in enumerate(selected, 1)
                ],
                "context_statement_count": len(selected),
                "context_word_count": len(context.split()),
                "retrieval_metrics": retrieval,
            }
            if model_ran:
                started = time.perf_counter()
                raw, usage = oracle.call_chat_completions(
                    args.base_url,
                    args.api_key,
                    args.model,
                    native.prompt(item["question"], context, rep),
                    args.timeout,
                    args.retries,
                    args.max_tokens,
                )
                row["latency_seconds"] = time.perf_counter() - started
                row["prediction"] = oracle.extract_choice(raw, item["choices"])
                row["raw_response"] = raw
                row["correct"] = oracle.normalized_equal(row["prediction"], item["answer"])
                row["usage"] = usage
            rows.append(row)
            print(f"{rep}: {number}/{len(questions)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle.write_jsonl(args.output_dir / "retrieval_results.jsonl", rows)
    if model_ran:
        oracle.write_jsonl(args.output_dir / "predictions.jsonl", rows)
    report = build_report(rows, model_ran, config)
    oracle.write_json(args.output_dir / "configuration.json", config)
    oracle.write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
