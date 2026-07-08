#!/usr/bin/env python3
"""E2: force complete Gold evidence into native E3 retrieval noise for 15 questions."""

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
RAG_DIR = DATASET / "rag_full_kb_processing"
sys.path.insert(0, str(ORACLE_DIR))

import run_oracle_20 as oracle  # noqa: E402
from build_pilot_representations import format_value, statement_to_sentence  # noqa: E402


REPS = ("document", "triple", "graph")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qualifier_text(qualifiers: list[dict[str, Any]]) -> str:
    if not qualifiers:
        return ""
    rendered = "; ".join(
        f"{item['key']}={format_value(item['value'])}" for item in qualifiers
    )
    return f" [qualifiers: {rendered}]"


def triple_line(statement: dict[str, Any]) -> str:
    return (
        f"({statement['subject']['label']} | {statement['predicate']} | "
        f"{statement['object']['label']}){qualifier_text(statement['qualifiers'])}"
    )


def render_context(statements: list[dict[str, Any]], rep: str) -> str:
    """Render one ordered statement sequence without changing its factual content."""
    if rep == "document":
        return " ".join(statement_to_sentence(item) for item in statements)
    if rep == "triple":
        return "\n".join(triple_line(item) for item in statements)

    edges = ["Directed labelled graph (one edge per line):"]
    for statement in statements:
        source_id = statement["subject"]["id"]
        source_label = statement["subject"]["label"]
        if statement["object"]["kind"] == "literal":
            target_id = f"literal_{statement['statement_id']}"
        else:
            target_id = statement["object"]["id"]
        target_label = statement["object"]["label"]
        edges.append(
            f"EDGE {source_id} ({source_label}) --[{statement['predicate']}]--> "
            f"{target_id} ({target_label})"
            + qualifier_text(statement["qualifiers"])
        )
    return "\n".join(edges)


def unique_retrieved_ids(row: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    """Flatten selected chunks in retrieval order and retain first-seen rank."""
    ids: list[str] = []
    first_rank: dict[str, int] = {}
    for chunk in sorted(row["selected_chunks"], key=lambda item: item["rank"]):
        for statement_id in chunk["statement_ids"]:
            if statement_id not in first_rank:
                first_rank[statement_id] = chunk["rank"]
                ids.append(statement_id)
    return ids, first_rank


def force_gold(row: dict[str, Any]) -> dict[str, Any]:
    """Replace lowest-ranked non-Gold statements one-for-one with missing Gold."""
    original_ids, first_rank = unique_retrieved_ids(row)
    gold_ids = list(row["gold_statement_ids"])
    gold_set = set(gold_ids)
    original_set = set(original_ids)
    missing = [statement_id for statement_id in gold_ids if statement_id not in original_set]

    removable = sorted(
        (statement_id for statement_id in original_ids if statement_id not in gold_set),
        key=lambda statement_id: (first_rank[statement_id], original_ids.index(statement_id)),
        reverse=True,
    )
    if len(removable) < len(missing):
        raise RuntimeError(f"Not enough noise statements to replace for {row['question_id']}")

    final_ids = list(original_ids)
    removed = removable[: len(missing)]
    replacements = []
    for missing_id, removed_id in zip(missing, removed):
        position = final_ids.index(removed_id)
        final_ids[position] = missing_id
        replacements.append(
            {
                "position": position,
                "removed_noise_statement_id": removed_id,
                "removed_noise_first_chunk_rank": first_rank[removed_id],
                "inserted_gold_statement_id": missing_id,
            }
        )

    if len(final_ids) != len(original_ids) or not gold_set <= set(final_ids):
        raise AssertionError(f"E2 construction invariant failed: {row['question_id']}")
    return {
        "original_statement_ids": original_ids,
        "original_statement_count": len(original_ids),
        "original_covered_gold_statement_ids": [sid for sid in gold_ids if sid in original_set],
        "original_missing_gold_statement_ids": missing,
        "removed_noise_statement_ids": removed,
        "replacements": replacements,
        "final_statement_ids": final_ids,
        "final_statement_count": len(final_ids),
        "final_complete_gold": True,
    }


def load_needed_statements(path: Path, needed_ids: set[str]) -> dict[str, dict[str, Any]]:
    statements: dict[str, dict[str, Any]] = {}
    for statement in read_jsonl(path):
        statement_id = statement["statement_id"]
        if statement_id in needed_ids:
            statements[statement_id] = statement
            if len(statements) == len(needed_ids):
                break
    missing = needed_ids - statements.keys()
    if missing:
        raise RuntimeError(f"Canonical statements missing {len(missing)} IDs")
    return statements


def build_bundle(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = {row["question_id"]: row for row in read_jsonl(args.questions)}
    retrieval_rows = [
        row
        for row in read_jsonl(args.retrieval_results)
        if row["representation"] in args.representations
    ]
    expected = {(qid, rep) for qid in questions for rep in args.representations}
    actual = {(row["question_id"], row["representation"]) for row in retrieval_rows}
    if actual != expected or len(retrieval_rows) != len(expected):
        raise RuntimeError(
            f"Expected exactly {len(expected)} question/representation rows; got "
            f"{len(retrieval_rows)} rows and {len(actual)} unique keys"
        )

    constructions = []
    needed_ids: set[str] = set()
    for row in retrieval_rows:
        built = force_gold(row)
        constructions.append((row, built))
        needed_ids.update(built["original_statement_ids"])
        needed_ids.update(built["final_statement_ids"])
    statements = load_needed_statements(args.canonical_statements, needed_ids)

    bundle = []
    for row, built in constructions:
        question = questions[row["question_id"]]
        ordered = [statements[statement_id] for statement_id in built["final_statement_ids"]]
        context = render_context(ordered, row["representation"])
        source_ordered = [
            statements[statement_id] for statement_id in built["original_statement_ids"]
        ]
        source_replay_context = render_context(source_ordered, row["representation"])
        bundle.append(
            {
                "question_id": row["question_id"],
                "train_index": question["train_index"],
                "hop_count": row["hop_count"],
                "representation": row["representation"],
                "question": row["question"],
                "choices": question["choices"],
                "answer": question["answer"],
                "gold_statement_ids": row["gold_statement_ids"],
                "source_retrieval_metrics": row["retrieval_metrics"],
                **built,
                "source_replay_context": source_replay_context,
                "source_replay_context_word_count": len(source_replay_context.split()),
                "context": context,
                "context_word_count": len(context.split()),
            }
        )
    return bundle


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]], model_ran: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": len(rows),
        "source_complete_path_recall": average(
            [float(row["source_retrieval_metrics"]["complete_path_recall"]) for row in rows]
        ),
        "final_complete_path_recall": average(
            [float(row["final_complete_gold"]) for row in rows]
        ),
        "mean_original_statements": average(
            [float(row["original_statement_count"]) for row in rows]
        ),
        "mean_final_statements": average([float(row["final_statement_count"]) for row in rows]),
        "mean_replacements": average([float(len(row["replacements"])) for row in rows]),
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


def build_metrics(rows: list[dict[str, Any]], model_ran: bool, model: str | None) -> dict:
    hops = sorted({row["hop_count"] for row in rows})
    representations = [rep for rep in REPS if any(row["representation"] == rep for row in rows)]
    return {
        "status": "completed" if model_ran else "build_only",
        "condition": "E2_forced_gold_native_rag_noise",
        "questions": len({row["question_id"] for row in rows}),
        "contexts": len(rows),
        "model_calls": len(rows) if model_ran else 0,
        "model": model if model_ran else None,
        "by_representation": {
            rep: summarize([row for row in rows if row["representation"] == rep], model_ran)
            for rep in representations
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
                for rep in representations
            }
            for hop in hops
        },
    }


def make_prompt(item: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Use only retrieved context. If the answer is absent, return UNKNOWN.",
        },
        {
            "role": "user",
            "content": (
                f"Knowledge representation: {item['representation']}\n\n"
                f"Retrieved context:\n{item['context']}\n\n"
                f"Question:\n{item['question']}\n\n"
                "Return only the exact answer entity or value."
            ),
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ORACLE_DIR / "output" / "oracle_hops_15" / "oracle_questions.jsonl",
    )
    parser.add_argument(
        "--retrieval-results",
        type=Path,
        default=RAG_DIR / "output" / "rag_hops_15" / "retrieval_results.jsonl",
    )
    parser.add_argument(
        "--canonical-statements",
        type=Path,
        default=RAG_DIR / "output" / "full_kb" / "canonical" / "statements.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "e2_hops_15")
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPS,
        default=list(REPS),
        help="Representations to run; use '--representations graph' for Graph only.",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_bundle(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle.write_jsonl(args.output_dir / "e2_contexts.jsonl", rows)

    validation = {
        "condition": "E2_forced_gold_native_rag_noise",
        "questions": len({row["question_id"] for row in rows}),
        "contexts": len(rows),
        "representations": list(args.representations),
        "graph_serialization": (
            "EDGE source_id (source_label) --[predicate]--> target_id (target_label)"
            if "graph" in args.representations
            else None
        ),
        "source": str(args.retrieval_results),
        "all_gold_present": all(row["final_complete_gold"] for row in rows),
        "statement_count_preserved": all(
            row["original_statement_count"] == row["final_statement_count"] for row in rows
        ),
        "gold_used_for_original_retrieval": False,
        "gold_used_only_after_retrieval": True,
        "native_noise_per_representation": True,
        "existing_experiments_modified": False,
    }
    oracle.write_json(args.output_dir / "bundle_validation.json", validation)

    model_ran = not args.build_only and bool(args.base_url and args.model)
    if model_ran:
        for number, item in enumerate(rows, 1):
            started = time.perf_counter()
            raw, usage = oracle.call_chat_completions(
                args.base_url,
                args.api_key,
                args.model,
                make_prompt(item),
                args.timeout,
                args.retries,
                args.max_tokens,
            )
            item["latency_seconds"] = time.perf_counter() - started
            item["prediction"] = oracle.extract_choice(raw, item["choices"])
            item["raw_response"] = raw
            item["correct"] = oracle.normalized_equal(item["prediction"], item["answer"])
            item["usage"] = usage
            print(f"Completed {number}/{len(rows)}", flush=True)
        oracle.write_jsonl(args.output_dir / "predictions.jsonl", rows)

    report = build_metrics(rows, model_ran, args.model)
    report["validation"] = validation
    oracle.write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
