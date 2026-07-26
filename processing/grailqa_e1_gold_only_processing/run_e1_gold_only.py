#!/usr/bin/env python3
"""Run GrailQA E1 with complete Gold evidence in three representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PROCESSING = HERE.parent
PROJECT = HERE.parent.parent
E0_DIR = PROCESSING / "grailqa_e0_processing"
sys.path.insert(0, str(E0_DIR))
import run_e0_closed_book as e0  # noqa: E402


DEFAULT_QUESTIONS = (
    PROCESSING
    / "grailqa_mini_kb_processing"
    / "output"
    / "dev_10_per_hop"
    / "selected_questions.jsonl"
)
DEFAULT_KB = (
    PROCESSING
    / "grailqa_mini_kb_processing"
    / "output"
    / "dev_10_per_hop"
    / "mini_kb.jsonl"
)
DEFAULT_ALIAS_CACHE = E0_DIR / "output" / "answer_alias_cache.json"
REPRESENTATIONS = ("document", "triple", "graph")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_questions(
    rows: list[dict[str, Any]], max_per_hop: int | None
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_hop: Counter[int] = Counter()
    for row in sorted(rows, key=lambda item: (item["hop_count"], item["dataset_index"])):
        hop = int(row["hop_count"])
        if max_per_hop is not None and per_hop[hop] >= max_per_hop:
            continue
        selected.append(row)
        per_hop[hop] += 1
    return selected


def render_context(
    statement_ids: list[str],
    kb_by_id: dict[str, dict[str, Any]],
    representation: str,
) -> str:
    lines = []
    for statement_id in statement_ids:
        statement = kb_by_id[statement_id]
        lines.append(f"[{statement_id}] {statement[representation]}")
    return "\n".join(lines)


def build_cases(
    questions: list[dict[str, Any]],
    kb_by_id: dict[str, dict[str, Any]],
    gold_context: str,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for question in questions:
        if gold_context == "sufficient":
            if "sufficient_gold_statement_ids" not in question:
                raise RuntimeError(
                    f"Question {question['question_id']} does not contain "
                    "sufficient_gold_statement_ids"
                )
            statement_ids = list(question["sufficient_gold_statement_ids"])
        else:
            statement_ids = list(question["gold_statement_ids"])
        missing = [statement_id for statement_id in statement_ids if statement_id not in kb_by_id]
        if missing:
            raise RuntimeError(
                f"Question {question['question_id']} has Gold IDs missing from mini-KB: {missing}"
            )
        contexts = {
            representation: render_context(statement_ids, kb_by_id, representation)
            for representation in REPRESENTATIONS
        }
        cases.append(
            {
                "question_id": str(question["question_id"]),
                "dataset_index": int(question["dataset_index"]),
                "hop_count": int(question["hop_count"]),
                "level": question.get("level"),
                "question": question["question"],
                "gold_answers": [
                    answer.get("entity_name") or answer["answer_argument"]
                    for answer in question["answers"]
                ],
                "gold_answer_ids": list(question["answer_ids"]),
                "gold_statement_ids": statement_ids,
                "gold_statement_count": len(statement_ids),
                "contexts": contexts,
                "context_sha256": {
                    representation: sha256_text(context)
                    for representation, context in contexts.items()
                },
            }
        )
    return cases


def gold_only_prompt(
    question: str, context: str, representation: str
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer using only the provided Gold knowledge context. The context "
                "contains all grounded facts selected for this question and contains no "
                "retrieval noise. A question may have one or multiple correct answers."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Knowledge representation: {representation}\n\n"
                f"Gold knowledge context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "Return the complete answer set as JSON. Use the exact format "
                '{"answers":["answer 1","answer 2"]}. Include no explanation.'
            ),
        },
    ]


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return e0.summarize(rows)


def metrics_report(
    predictions: list[dict[str, Any]],
    model: str,
    validation: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    hops = sorted({row["hop_count"] for row in predictions})
    return {
        "status": "completed",
        "metrics_schema_version": 2,
        "condition": "E1_grailqa_gold_only",
        "model": model,
        "questions": validation["questions"],
        "calls": len(predictions),
        "evaluation": e0.evaluation_metadata(configuration["alias_cache"]),
        "configuration": configuration,
        "by_representation": {
            representation: summarize_predictions(
                [row for row in predictions if row["representation"] == representation]
            )
            for representation in REPRESENTATIONS
        },
        "by_hop": {
            str(hop): {
                "questions": validation["by_hop"][str(hop)],
                "by_representation": {
                    representation: summarize_predictions(
                        [
                            row
                            for row in predictions
                            if row["hop_count"] == hop
                            and row["representation"] == representation
                        ]
                    )
                    for representation in REPRESENTATIONS
                },
            }
            for hop in hops
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB)
    parser.add_argument("--alias-cache", type=Path, default=DEFAULT_ALIAS_CACHE)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "output" / "e1_gold_30"
    )
    parser.add_argument("--max-per-hop", type=int)
    parser.add_argument(
        "--gold-context",
        choices=["materialized", "sufficient"],
        default="materialized",
        help=(
            "Use every materialized SPARQL edge or the deterministic minimal "
            "proof union persisted with each question."
        ),
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3-4b-16k"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.questions, args.kb, args.alias_cache):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.max_per_hop is not None and args.max_per_hop < 1:
        raise ValueError("--max-per-hop must be positive")

    source_questions = read_jsonl(args.questions)
    selected = select_questions(source_questions, args.max_per_hop)
    kb_rows = read_jsonl(args.kb)
    kb_by_id = {row["statement_id"]: row for row in kb_rows}
    if len(kb_by_id) != len(kb_rows):
        raise RuntimeError("mini_kb.jsonl contains duplicate statement IDs")
    cases = build_cases(selected, kb_by_id, args.gold_context)
    alias_cache, alias_metadata = e0.load_alias_cache(args.alias_cache)
    required_gold_ids = {mid for case in cases for mid in case["gold_answer_ids"]}
    missing_alias_ids = sorted(required_gold_ids - set(alias_cache))
    if missing_alias_ids:
        raise RuntimeError(f"Alias cache misses Gold MIDs: {missing_alias_ids[:20]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "e1_cases.jsonl", cases)
    hops = sorted({case["hop_count"] for case in cases})
    context_lengths = {
        representation: [len(case["contexts"][representation]) for case in cases]
        for representation in REPRESENTATIONS
    }
    validation = {
        "condition": "E1_grailqa_gold_only",
        "source_questions": str(args.questions.resolve()),
        "source_mini_kb": str(args.kb.resolve()),
        "questions": len(cases),
        "cases": len(cases) * len(REPRESENTATIONS),
        "by_hop": {
            str(hop): sum(case["hop_count"] == hop for case in cases) for hop in hops
        },
        "representations": list(REPRESENTATIONS),
        "gold_context_only": True,
        "gold_context_definition": args.gold_context,
        "complete_gold_loaded": all(case["gold_statement_count"] > 0 for case in cases),
        "noise_statement_count": 0,
        "retrieval_used": False,
        "statement_ids_shared_across_representations": True,
        "statement_order_shared_across_representations": True,
        "min_gold_statement_count": min(case["gold_statement_count"] for case in cases),
        "max_gold_statement_count": max(case["gold_statement_count"] for case in cases),
        "max_context_characters": {
            representation: max(lengths)
            for representation, lengths in context_lengths.items()
        },
        "alias_cache": alias_metadata,
        "alias_cache_loaded_into_prompt": False,
        "missing_alias_gold_mids": missing_alias_ids,
    }
    write_json(args.output_dir / "bundle_validation.json", validation)
    if args.build_only:
        report = {"status": "build_only", **validation}
        write_json(args.output_dir / "metrics.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    predictions: list[dict[str, Any]] = []
    total_calls = len(cases) * len(REPRESENTATIONS)
    call_number = 0
    for case in cases:
        for representation in REPRESENTATIONS:
            call_number += 1
            started = time.perf_counter()
            raw, usage = e0.call_ollama(
                args.base_url,
                args.model,
                gold_only_prompt(
                    case["question"], case["contexts"][representation], representation
                ),
                args.timeout,
                args.retries,
                args.max_tokens,
            )
            latency = time.perf_counter() - started
            predicted, valid, parse_error = e0.parse_answers(raw)
            predictions.append(
                {
                    "question_id": case["question_id"],
                    "dataset_index": case["dataset_index"],
                    "hop_count": case["hop_count"],
                    "level": case["level"],
                    "representation": representation,
                    "question": case["question"],
                    "gold_answers": case["gold_answers"],
                    "gold_answer_ids": case["gold_answer_ids"],
                    "gold_statement_ids": case["gold_statement_ids"],
                    "gold_statement_count": case["gold_statement_count"],
                    "context_sha256": case["context_sha256"][representation],
                    "prediction": predicted,
                    "raw_response": raw,
                    "output_valid": valid,
                    "parse_error": parse_error,
                    **e0.score_canonical_answer_set(predicted, case["gold_answers"]),
                    **e0.score_alias_answer_set(
                        predicted,
                        case["gold_answers"],
                        case["gold_answer_ids"],
                        alias_cache,
                    ),
                    "usage": usage,
                    "latency_seconds": latency,
                }
            )
            write_jsonl(args.output_dir / "predictions.jsonl", predictions)
            print(
                f"Completed {call_number}/{total_calls} "
                f"({case['hop_count']}-hop, {representation})",
                flush=True,
            )

    configuration = {
        "gold_context_only": True,
        "gold_context_definition": args.gold_context,
        "noise_statement_count": 0,
        "retrieval_used": False,
        "temperature": 0,
        "think": False,
        "structured_json": True,
        "max_tokens": args.max_tokens,
        "maximum_answer_string_characters": 160,
        "maximum_answer_count": 128,
        "representations": list(REPRESENTATIONS),
        "alias_cache": alias_metadata,
        "alias_cache_loaded_into_prompt": False,
    }
    report = metrics_report(predictions, args.model, validation, configuration)
    write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
