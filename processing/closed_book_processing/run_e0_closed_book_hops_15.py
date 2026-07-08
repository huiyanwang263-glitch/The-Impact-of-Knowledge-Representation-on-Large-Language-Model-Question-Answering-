#!/usr/bin/env python3
"""E0 closed-book baseline on the fixed 1/2/3-hop KQA Pro questions."""

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


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def closed_book_prompt(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the question from your own internal knowledge. "
                "No external knowledge or retrieved context is provided."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                "Return only the exact answer entity or value, with no explanation."
            ),
        },
    ]


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def usage_value(row: dict[str, Any], key: str) -> int:
    value = row.get("usage", {}).get(key)
    return int(value) if isinstance(value, (int, float)) else 0


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in rows)
    unmatched = sum(row["prediction"] is None for row in rows)
    matched = len(rows) - unmatched
    prompt_tokens = [usage_value(row, "prompt_tokens") for row in rows]
    completion_tokens = [usage_value(row, "completion_tokens") for row in rows]
    total_tokens = [usage_value(row, "total_tokens") for row in rows]
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows) if rows else None,
        "unmatched_outputs": unmatched,
        "unmatched_rate": unmatched / len(rows) if rows else None,
        "accuracy_on_matched_outputs": correct / matched if matched else None,
        "mean_prompt_tokens": average([float(value) for value in prompt_tokens]),
        "mean_completion_tokens": average([float(value) for value in completion_tokens]),
        "mean_total_tokens": average([float(value) for value in total_tokens]),
        "total_prompt_tokens": sum(prompt_tokens),
        "total_completion_tokens": sum(completion_tokens),
        "total_tokens": sum(total_tokens),
        "mean_model_latency_ms": average([row["model_latency_ms"] for row in rows]),
        "total_model_latency_ms": sum(row["model_latency_ms"] for row in rows),
        "tokens_per_correct_answer": sum(total_tokens) / correct if correct else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ORACLE_DIR / "output" / "oracle_hops_15" / "oracle_questions.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "e0_hops_15")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = list(read_jsonl(args.questions))
    questions = [
        {
            "question_id": row["question_id"],
            "train_index": row["train_index"],
            "hop_count": row["hop_count"],
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
        }
        for row in source_rows
    ]
    if not questions:
        raise RuntimeError(f"No questions found in {args.questions}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle.write_jsonl(args.output_dir / "closed_book_questions.jsonl", questions)
    validation = {
        "condition": "E0_closed_book",
        "questions": len(questions),
        "by_hop": {
            str(hop): sum(row["hop_count"] == hop for row in questions)
            for hop in sorted({row["hop_count"] for row in questions})
        },
        "knowledge_context_provided": False,
        "retrieval_used": False,
        "gold_statements_loaded_into_prompt": False,
        "choices_shown_to_model": False,
        "existing_experiments_modified": False,
    }
    oracle.write_json(args.output_dir / "bundle_validation.json", validation)

    if args.build_only or not (args.base_url and args.model):
        report = {"status": "build_only", "llm_accuracy": None, **validation}
        oracle.write_json(args.output_dir / "metrics.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    predictions = []
    for number, item in enumerate(questions, start=1):
        start = time.perf_counter()
        raw, usage = oracle.call_chat_completions(
            args.base_url,
            args.api_key,
            args.model,
            closed_book_prompt(item["question"]),
            args.timeout,
            args.retries,
            args.max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        prediction = oracle.extract_choice(raw, item["choices"])
        predictions.append(
            {
                "question_id": item["question_id"],
                "hop_count": item["hop_count"],
                "gold_answer": item["answer"],
                "prediction": prediction,
                "raw_response": raw,
                "correct": oracle.normalized_equal(prediction, item["answer"]),
                "usage": usage,
                "model_latency_ms": latency_ms,
            }
        )
        print(f"Completed {number}/{len(questions)} ({item['hop_count']}-hop)", flush=True)

    hops = sorted({item["hop_count"] for item in questions})
    report = {
        "status": "completed",
        "condition": "E0_closed_book",
        "model": args.model,
        "questions": len(questions),
        "calls": len(predictions),
        "configuration": {
            "knowledge_context_provided": False,
            "retrieval_used": False,
            "choices_shown_to_model": False,
            "temperature": 0,
            "max_tokens": args.max_tokens,
        },
        "overall": summarise(predictions),
        "by_hop": {
            str(hop): summarise(
                [row for row in predictions if row["hop_count"] == hop]
            )
            for hop in hops
        },
    }
    oracle.write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    oracle.write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
