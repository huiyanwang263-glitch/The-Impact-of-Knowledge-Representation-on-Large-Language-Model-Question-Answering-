#!/usr/bin/env python3
"""Same-question RAG test with one shared BM25 procedure for all representations."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DATASET = HERE.parent
ORACLE_DIR = DATASET / "knowledge_representation_processing"
sys.path.insert(0, str(ORACLE_DIR))
import run_oracle_20 as oracle  # noqa: E402


REPS = ("document", "triple", "graph")
WORD_RE = re.compile(r"(?u)\b\w+\b")


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def row_statement_ids(row: dict[str, Any], rep: str) -> list[str]:
    return [row["statement_id"]] if rep == "triple" else list(row["statement_ids"])


class BM25:
    """Query-focused BM25: exact scores, but stores only rows sharing a query term."""

    def __init__(self, path: Path, rep: str, vocabulary: set[str], k1: float, b: float):
        self.rep, self.k1, self.b = rep, k1, b
        self.n, total = 0, 0
        self.df: Counter[str] = Counter()
        self.rows: list[dict[str, Any]] = []
        for row in read_jsonl(path):
            terms = tokens(row["text"])
            self.n += 1
            total += len(terms)
            tf = Counter(term for term in terms if term in vocabulary)
            self.df.update(tf.keys())
            if tf:
                self.rows.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "text": row["text"],
                        "statement_ids": row_statement_ids(row, rep),
                        "length": len(terms),
                        "tf": tf,
                    }
                )
        if not self.n:
            raise RuntimeError(f"Empty corpus: {path}")
        self.avgdl = total / self.n

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        query_terms = list(dict.fromkeys(tokens(query)))
        ranked = []
        for row in self.rows:
            score = 0.0
            for term in query_terms:
                tf = row["tf"].get(term, 0)
                if not tf:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                norm = tf + self.k1 * (1 - self.b + self.b * row["length"] / self.avgdl)
                score += idf * tf * (self.k1 + 1) / norm
            if score:
                ranked.append({**row, "score": score})
        ranked.sort(key=lambda row: (-row["score"], row["chunk_id"]))
        return ranked[:limit]


def choose_context(ranked: list[dict[str, Any]], budget: int) -> tuple[list, str, int]:
    chosen, used = [], 0
    for row in ranked:
        if used + row["length"] <= budget:
            chosen.append(row)
            used += row["length"]
    return chosen, "\n\n".join(row["text"] for row in chosen), used


def retrieval_scores(ranked: list, chosen: list, gold_ids: list[str]) -> dict[str, Any]:
    gold = set(gold_ids)
    retrieved = {sid for row in chosen for sid in row["statement_ids"]}
    covered = gold & retrieved
    first_rank = next(
        (i for i, row in enumerate(ranked, 1) if gold & set(row["statement_ids"])), None
    )
    return {
        "gold_statement_count": len(gold),
        "retrieved_statement_count": len(retrieved),
        "covered_gold_statement_ids": sorted(covered),
        "statement_recall": len(covered) / len(gold),
        "hit": bool(covered),
        "complete_path_recall": gold <= retrieved,
        "first_gold_rank": first_rank,
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
    }


def prompt(question: str, context: str, rep: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Use only retrieved context. If the answer is absent, return UNKNOWN.",
        },
        {
            "role": "user",
            "content": (
                f"Knowledge representation: {rep}\n\nRetrieved context:\n{context}\n\n"
                f"Question:\n{question}\n\nReturn only the exact answer entity or value."
            ),
        },
    ]


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summary(rows: list[dict[str, Any]], with_answers: bool) -> dict[str, Any]:
    result = {
        "questions": len(rows),
        "mean_statement_recall": average(
            [row["retrieval_metrics"]["statement_recall"] for row in rows]
        ),
        "hit_rate": average([float(row["retrieval_metrics"]["hit"]) for row in rows]),
        "complete_path_recall": average(
            [float(row["retrieval_metrics"]["complete_path_recall"]) for row in rows]
        ),
        "mrr": average([row["retrieval_metrics"]["reciprocal_rank"] for row in rows]),
        "mean_context_words": average([float(row["context_word_count"]) for row in rows]),
    }
    if with_answers:
        correct = sum(row["correct"] for row in rows)
        invalid = sum(row["prediction"] is None for row in rows)
        valid = len(rows) - invalid
        result["answer"] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows),
            "invalid_outputs": invalid,
            "accuracy_on_valid_outputs": correct / valid if valid else None,
        }
    return result


def metrics(rows: list[dict[str, Any]], model_ran: bool, config: dict) -> dict:
    hops = sorted({row["hop_count"] for row in rows})
    return {
        "status": "completed" if model_ran else "retrieval_only",
        "questions": len({row["question_id"] for row in rows}),
        "retrieval_runs": len(rows),
        "model_calls": len(rows) if model_ran else 0,
        "configuration": config,
        "by_representation": {
            rep: summary([row for row in rows if row["representation"] == rep], model_ran)
            for rep in REPS
        },
        "by_hop": {
            str(hop): {
                rep: summary(
                    [
                        row
                        for row in rows
                        if row["hop_count"] == hop and row["representation"] == rep
                    ],
                    model_ran,
                )
                for rep in REPS
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
    parser.add_argument("--corpus-root", type=Path, default=HERE / "output" / "full_kb")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "rag_hops_15")
    parser.add_argument("--context-word-budget", type=int, default=2000)
    parser.add_argument("--max-ranked-chunks", type=int, default=1000)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = list(read_jsonl(args.questions))
    if not questions:
        raise RuntimeError(f"No questions: {args.questions}")
    vocabulary = {token for item in questions for token in tokens(item["question"])}
    paths = {
        "document": args.corpus_root / "document" / "documents.jsonl",
        "triple": args.corpus_root / "triple" / "unordered_triples.jsonl",
        "graph": args.corpus_root / "graph" / "adjacency_chunks.jsonl",
    }
    model_ran = not args.retrieval_only and bool(args.base_url and args.model)
    config = {
        "retriever": "BM25",
        "same_retrieval_method_for_all_representations": True,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "tokenizer": "Unicode word regex + casefold",
        "query": "original question only",
        "context_word_budget": args.context_word_budget,
        "max_ranked_chunks": args.max_ranked_chunks,
        "tie_break": "chunk_id ascending",
        "gold_used_during_retrieval": False,
        "graph_serialization": (
            "EDGE source_id (source_label) --[predicate]--> target_id (target_label)"
        ),
        "model": args.model if model_ran else None,
    }

    output_rows: list[dict[str, Any]] = []
    for rep in REPS:
        print(f"Building BM25 for {rep}...", flush=True)
        retriever = BM25(paths[rep], rep, vocabulary, args.bm25_k1, args.bm25_b)
        for number, item in enumerate(questions, 1):
            ranked = retriever.search(item["question"], args.max_ranked_chunks)
            chosen, context, used = choose_context(ranked, args.context_word_budget)
            retrieval = retrieval_scores(ranked, chosen, item["gold_statement_ids"])
            row: dict[str, Any] = {
                "question_id": item["question_id"],
                "hop_count": item["hop_count"],
                "representation": rep,
                "question": item["question"],
                "gold_answer": item["answer"],
                "gold_statement_ids": item["gold_statement_ids"],
                "selected_chunks": [
                    {
                        "rank": ranked.index(chunk) + 1,
                        "chunk_id": chunk["chunk_id"],
                        "score": chunk["score"],
                        "statement_ids": chunk["statement_ids"],
                        "word_count": chunk["length"],
                        "text": chunk["text"],
                    }
                    for chunk in chosen
                ],
                "context_word_count": used,
                "retrieval_metrics": retrieval,
            }
            if model_ran:
                raw, usage = oracle.call_chat_completions(
                    args.base_url,
                    args.api_key,
                    args.model,
                    prompt(item["question"], context, rep),
                    args.timeout,
                    args.retries,
                    args.max_tokens,
                )
                prediction = oracle.extract_choice(raw, item["choices"])
                row.update(
                    prediction=prediction,
                    raw_response=raw,
                    correct=oracle.normalized_equal(prediction, item["answer"]),
                    usage=usage,
                )
            output_rows.append(row)
            print(f"{rep}: {number}/{len(questions)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle.write_jsonl(args.output_dir / "retrieval_results.jsonl", output_rows)
    if model_ran:
        oracle.write_jsonl(
            args.output_dir / "predictions.jsonl",
            [
                {key: row[key] for key in (
                    "question_id", "hop_count", "representation", "gold_answer",
                    "prediction", "raw_response", "correct", "usage"
                )}
                for row in output_rows
            ],
        )
    report = metrics(output_rows, model_ran, config)
    oracle.write_json(args.output_dir / "metrics.json", report)
    oracle.write_json(args.output_dir / "configuration.json", config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
