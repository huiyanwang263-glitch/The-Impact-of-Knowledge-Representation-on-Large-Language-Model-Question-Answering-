#!/usr/bin/env python3
"""KQA Pro E2: repair saved E3-M selections with complete Gold evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROCESSING = HERE.parent
ORACLE_DIR = PROCESSING / "knowledge_representation_processing"
E3_DIR = PROCESSING / "e3_manual_llm_retrieval_processing"
DEFAULT_E3_OUTPUT = E3_DIR / "output" / "e3_manual_hops_15"
sys.path.insert(0, str(E3_DIR))
import run_e3_manual_llm_retrieval_hops_15 as e3  # noqa: E402


REPS = e3.REPS
PROMPT_VERSION = "e3_manual_answer_prompt_v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def select_questions(
    questions: list[dict[str, Any]], max_per_hop: int | None
) -> list[dict[str, Any]]:
    if max_per_hop is None:
        return list(questions)
    selected: list[dict[str, Any]] = []
    counts: Counter[int] = Counter()
    for question in questions:
        hop = int(question["hop_count"])
        if counts[hop] >= max_per_hop:
            continue
        selected.append(question)
        counts[hop] += 1
    return selected


def force_complete_gold(
    selected_ids: list[str], gold_ids: list[str]
) -> tuple[list[str], dict[str, Any]]:
    """Replace tail noise with missing Gold, expanding only when unavoidable."""
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("E3 selection contains duplicate statement IDs")
    if len(gold_ids) != len(set(gold_ids)):
        raise RuntimeError("Gold statement IDs are not unique")

    gold_set = set(gold_ids)
    selected_set = set(selected_ids)
    missing_gold = [statement_id for statement_id in gold_ids if statement_id not in selected_set]
    final_ids = list(selected_ids)
    removable_positions = [
        index
        for index in range(len(final_ids) - 1, -1, -1)
        if final_ids[index] not in gold_set
    ]
    replacements: list[dict[str, Any]] = []
    appended_gold: list[str] = []
    removed_noise: list[str] = []
    for statement_id in missing_gold:
        if removable_positions:
            position = removable_positions.pop(0)
            removed_id = final_ids[position]
            final_ids[position] = statement_id
            removed_noise.append(removed_id)
            replacements.append(
                {
                    "position": position,
                    "removed_noise_statement_id": removed_id,
                    "inserted_gold_statement_id": statement_id,
                }
            )
        else:
            final_ids.append(statement_id)
            appended_gold.append(statement_id)

    if not gold_set <= set(final_ids):
        raise RuntimeError("Forced-Gold completion failed")
    if len(final_ids) != max(len(selected_ids), len(gold_ids)):
        raise RuntimeError("Forced-Gold context-size invariant failed")

    retained_noise = [statement_id for statement_id in final_ids if statement_id not in gold_set]
    return final_ids, {
        "raw_selected_statement_count": len(selected_ids),
        "forced_context_statement_count": len(final_ids),
        "missing_gold_statement_ids": missing_gold,
        "gold_added_statement_count": len(missing_gold),
        "replacements": replacements,
        "removed_noise_statement_ids": removed_noise,
        "removed_noise_statement_count": len(removed_noise),
        "appended_gold_statement_ids": appended_gold,
        "appended_gold_statement_count": len(appended_gold),
        "retained_noise_statement_ids": retained_noise,
        "retained_noise_statement_count": len(retained_noise),
        "context_size_preserved": len(final_ids) == len(selected_ids),
        "context_expanded": len(final_ids) > len(selected_ids),
    }


def selection_hash(rows: list[dict[str, Any]]) -> str:
    return e3.stable_digest(
        [
            {
                "question_id": str(row["question_id"]),
                "representation": row["representation"],
                "selected_statement_ids": row.get("selected_statement_ids", []),
                "selector_valid": bool(row.get("selector_valid")),
            }
            for row in sorted(
                rows,
                key=lambda item: (
                    str(item["question_id"]),
                    REPS.index(item["representation"]),
                ),
            )
        ]
    )


def build_bundle(
    questions: list[dict[str, Any]],
    candidate_banks: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    representations: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    question_by_id = {str(row["question_id"]): row for row in questions}
    candidate_by_id = {str(row["question_id"]): row for row in candidate_banks}
    selection_by_key = {
        (str(row["question_id"]), row["representation"]): row
        for row in selection_rows
        if row["representation"] in representations
    }
    if len(question_by_id) != len(questions):
        raise RuntimeError("Question IDs are not unique")
    expected = {
        (question_id, representation)
        for question_id in question_by_id
        for representation in representations
    }
    if set(selection_by_key) != expected:
        missing = sorted(expected - set(selection_by_key))
        extra = sorted(set(selection_by_key) - expected)
        raise RuntimeError(f"E3 selection case mismatch; missing={missing[:5]}, extra={extra[:5]}")

    result: list[dict[str, Any]] = []
    for question in questions:
        question_id = str(question["question_id"])
        if question_id not in candidate_by_id:
            raise RuntimeError(f"E3 candidate bank missing question {question_id}")
        bank = candidate_by_id[question_id]
        if list(question["gold_statement_ids"]) != list(bank["gold_statement_ids"]):
            raise RuntimeError(f"Question/candidate Gold mismatch for {question_id}")
        statement_by_id = {
            statement["statement_id"]: statement
            for statement in bank["candidate_statements"]
        }
        if set(statement_by_id) != set(bank["candidate_statement_ids"]):
            raise RuntimeError(f"Candidate statement mismatch for {question_id}")

        for representation in representations:
            source = selection_by_key[(question_id, representation)]
            if list(source.get("candidate_statement_ids", [])) != list(
                bank["candidate_statement_ids"]
            ):
                raise RuntimeError(
                    f"E3 selection/candidate order mismatch for {question_id}/{representation}"
                )
            raw_selected = list(source.get("selected_statement_ids", []))
            if len(raw_selected) > top_k:
                raise RuntimeError(f"E3 selection exceeds Top-K for {question_id}/{representation}")
            unknown = [statement_id for statement_id in raw_selected if statement_id not in statement_by_id]
            if unknown:
                raise RuntimeError(
                    f"E3 selection has unknown IDs for {question_id}/{representation}: {unknown[:5]}"
                )
            final_ids, forcing = force_complete_gold(
                raw_selected, list(bank["gold_statement_ids"])
            )
            forcing["invalid_selector_fallback_used"] = not bool(
                source.get("selector_valid")
            )
            if len(final_ids) > top_k:
                raise RuntimeError(f"Forced E2 context exceeds Top-K for {question_id}/{representation}")
            final_statements = [statement_by_id[statement_id] for statement_id in final_ids]
            context = e3.render_statements(final_statements, representation)
            result.append(
                {
                    "question_id": question_id,
                    "train_index": question["train_index"],
                    "hop_count": int(question["hop_count"]),
                    "representation": representation,
                    "question": question["question"],
                    "choices": question["choices"],
                    "answer": question["answer"],
                    "gold_statement_ids": list(bank["gold_statement_ids"]),
                    "candidate_statement_ids": list(bank["candidate_statement_ids"]),
                    "raw_selected_statement_ids": raw_selected,
                    "selected_statement_ids": final_ids,
                    "selected_statements": final_statements,
                    "selected_context": context,
                    "forcing": forcing,
                    "selector_source": "saved_e3_manual_selection",
                    "selector_valid": bool(source.get("selector_valid")),
                    "selector_attempts": source.get("selector_attempts"),
                    "selector_unknown_ids": source.get("selector_unknown_ids", []),
                    "selector_usage": source.get("selector_usage", {}),
                    "selector_latency_seconds": source.get("selector_latency_seconds", 0.0),
                    "retrieval_metrics_before_forcing": e3.retrieval_metrics(
                        raw_selected, list(bank["gold_statement_ids"])
                    ),
                    "retrieval_metrics": e3.retrieval_metrics(
                        final_ids, list(bank["gold_statement_ids"])
                    ),
                    "answer_attempted": False,
                    "prediction": None,
                    "correct": False,
                }
            )
    return result


def summarize(rows: list[dict[str, Any]], answer_ran: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cases": len(rows),
        "selector_before_forcing": {
            "selector_valid_rate": average([float(row["selector_valid"]) for row in rows]),
            "mean_statement_recall_at_k": average(
                [row["retrieval_metrics_before_forcing"]["statement_recall_at_k"] for row in rows]
            ),
            "hit_rate_at_k": average(
                [float(row["retrieval_metrics_before_forcing"]["hit_at_k"]) for row in rows]
            ),
            "complete_path_recall_at_k": average(
                [
                    float(row["retrieval_metrics_before_forcing"]["complete_path_recall_at_k"])
                    for row in rows
                ]
            ),
            "mean_selected_statement_count": average(
                [
                    float(row["retrieval_metrics_before_forcing"]["selected_statement_count"])
                    for row in rows
                ]
            ),
        },
        "forced_context": {
            "mean_statement_recall": average(
                [row["retrieval_metrics"]["statement_recall_at_k"] for row in rows]
            ),
            "complete_gold_rate": average(
                [float(row["retrieval_metrics"]["complete_path_recall_at_k"]) for row in rows]
            ),
            "mean_statement_count": average(
                [float(row["retrieval_metrics"]["selected_statement_count"]) for row in rows]
            ),
            "mean_gold_added": average(
                [float(row["forcing"]["gold_added_statement_count"]) for row in rows]
            ),
            "mean_noise_removed": average(
                [float(row["forcing"]["removed_noise_statement_count"]) for row in rows]
            ),
            "context_size_preserved_rate": average(
                [float(row["forcing"]["context_size_preserved"]) for row in rows]
            ),
            "context_expansion_rate": average(
                [float(row["forcing"]["context_expanded"]) for row in rows]
            ),
            "invalid_selector_fallback_rate": average(
                [float(row["forcing"]["invalid_selector_fallback_used"]) for row in rows]
            ),
        },
    }
    if answer_ran:
        correct = sum(bool(row["correct"]) for row in rows)
        invalid = sum(row["prediction"] is None for row in rows)
        result["answer"] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows) if rows else None,
            "invalid_outputs": invalid,
            "invalid_rate": invalid / len(rows) if rows else None,
        }
        result["answer_efficiency"] = {
            "mean_latency_seconds": average(
                [float(row.get("answer_latency_seconds", 0.0)) for row in rows]
            ),
            "mean_prompt_tokens": average(
                [float(row.get("answer_usage", {}).get("prompt_tokens", 0)) for row in rows]
            ),
            "mean_completion_tokens": average(
                [
                    float(row.get("answer_usage", {}).get("completion_tokens", 0))
                    for row in rows
                ]
            ),
            "mean_total_tokens": average(
                [float(row.get("answer_usage", {}).get("total_tokens", 0)) for row in rows]
            ),
        }
    return result


def build_report(
    rows: list[dict[str, Any]], configuration: dict[str, Any], answer_ran: bool
) -> dict[str, Any]:
    hops = sorted({int(row["hop_count"]) for row in rows})
    representations = [rep for rep in REPS if any(row["representation"] == rep for row in rows)]
    return {
        "status": "completed" if answer_ran else "build_only",
        "condition": "E2_manual_noise_forced_gold",
        "questions": len({row["question_id"] for row in rows}),
        "cases": len(rows),
        "new_selector_calls": 0,
        "reused_e3_selections": len(rows),
        "answer_calls": len(rows) if answer_ran else 0,
        "configuration": configuration,
        "by_representation": {
            rep: summarize([row for row in rows if row["representation"] == rep], answer_ran)
            for rep in representations
        },
        "by_hop": {
            str(hop): {
                rep: summarize(
                    [
                        row
                        for row in rows
                        if int(row["hop_count"]) == hop and row["representation"] == rep
                    ],
                    answer_ran,
                )
                for rep in representations
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
    parser.add_argument(
        "--candidate-banks",
        type=Path,
        default=DEFAULT_E3_OUTPUT / "candidate_banks.jsonl",
    )
    parser.add_argument(
        "--selection-results",
        type=Path,
        default=DEFAULT_E3_OUTPUT / "selection_results.jsonl",
    )
    parser.add_argument(
        "--e3-configuration",
        type=Path,
        default=DEFAULT_E3_OUTPUT / "configuration.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "output" / "e2_manual_hops_15"
    )
    parser.add_argument("--representations", nargs="+", choices=REPS, default=list(REPS))
    parser.add_argument("--max-per-hop", type=int)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--answer-max-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.questions,
        args.candidate_banks,
        args.selection_results,
        args.e3_configuration,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.max_per_hop is not None and args.max_per_hop < 1:
        raise ValueError("--max-per-hop must be positive")

    all_questions = list(e3.read_jsonl(args.questions))
    questions = select_questions(all_questions, args.max_per_hop)
    selected_question_ids = {str(row["question_id"]) for row in questions}
    candidate_banks = [
        row
        for row in e3.read_jsonl(args.candidate_banks)
        if str(row["question_id"]) in selected_question_ids
    ]
    all_selection_rows = list(e3.read_jsonl(args.selection_results))
    selection_rows = [
        row
        for row in all_selection_rows
        if str(row["question_id"]) in selected_question_ids
        and row["representation"] in args.representations
    ]
    e3_config = read_json(args.e3_configuration)
    top_k = int(e3_config["selected_top_k"])
    if e3_config.get("representations") and not set(args.representations) <= set(
        e3_config["representations"]
    ):
        raise RuntimeError("Requested representations are absent from E3 configuration")

    full_candidate_banks = list(e3.read_jsonl(args.candidate_banks))
    candidate_hash = e3.stable_digest(
        [
            {"question_id": row["question_id"], "ids": row["candidate_statement_ids"]}
            for row in full_candidate_banks
        ]
    )
    if candidate_hash != e3_config.get("candidate_bundle_sha256"):
        raise RuntimeError(
            "E3 candidate bundle hash mismatch: "
            f"computed={candidate_hash}, configured={e3_config.get('candidate_bundle_sha256')}"
        )
    selected_selection_hash = selection_hash(selection_rows)
    rows = build_bundle(
        questions,
        candidate_banks,
        selection_rows,
        list(args.representations),
        top_k,
    )

    configuration = {
        "source_condition": e3_config.get("condition"),
        "source_candidate_banks": str(args.candidate_banks.resolve()),
        "source_selection_results": str(args.selection_results.resolve()),
        "candidate_bundle_sha256": candidate_hash,
        "selected_selection_sha256": selected_selection_hash,
        "source_top_k": top_k,
        "selector_policy": "reuse_saved_e3_manual_selection",
        "forced_context_size_rule": "max(raw_selected_count, gold_statement_count)",
        "invalid_selector_fallback": "gold_only",
        "prompt_version": PROMPT_VERSION,
        "answer_prompt_shared_with_e3": True,
        "representations": list(args.representations),
        "max_per_hop": args.max_per_hop,
        "model": args.model or None,
        "answer_max_tokens": args.answer_max_tokens,
        "temperature": 0,
    }
    validation = {
        "condition": "E2_manual_noise_forced_gold",
        "questions": len(questions),
        "cases": len(rows),
        "by_hop": dict(Counter(str(row["hop_count"]) for row in questions)),
        "candidate_bundle_hash_matches_e3": True,
        "selection_reused_from_e3": True,
        "new_selector_calls": 0,
        "all_gold_present_after_forcing": all(
            row["retrieval_metrics"]["complete_path_recall_at_k"] for row in rows
        ),
        "all_forced_contexts_within_e3_top_k": all(
            len(row["selected_statement_ids"]) <= top_k for row in rows
        ),
        "candidate_ids_and_order_shared_across_representations": True,
        "answer_prompt_shared_with_e3": True,
    }
    if not all(
        value is True
        for key, value in validation.items()
        if key
        not in {"questions", "cases", "by_hop", "new_selector_calls", "condition"}
    ):
        raise RuntimeError(f"E2 validation failed: {validation}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    e3.write_jsonl(args.output_dir / "e2_contexts.jsonl", rows)
    e3.write_json(args.output_dir / "configuration.json", configuration)
    e3.write_json(args.output_dir / "bundle_validation.json", validation)
    model_configured = bool(args.base_url and args.model)
    if args.build_only or not model_configured:
        report = build_report(rows, configuration, answer_ran=False)
        report["validation"] = validation
        e3.write_json(args.output_dir / "metrics.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    run_signature = e3.stable_digest(
        {
            "candidate_bundle_sha256": candidate_hash,
            "selected_selection_sha256": selected_selection_hash,
            "model": args.model,
            "representations": args.representations,
            "max_per_hop": args.max_per_hop,
            "answer_max_tokens": args.answer_max_tokens,
            "prompt_version": PROMPT_VERSION,
        }
    )
    checkpoint_path = args.output_dir / "e2_results_checkpoint.jsonl"
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if checkpoint_path.exists():
        for saved in e3.read_jsonl(checkpoint_path):
            if saved.get("run_signature") == run_signature:
                completed[(str(saved["question_id"]), saved["representation"])] = saved

    for number, row in enumerate(rows, 1):
        key = (str(row["question_id"]), row["representation"])
        if key in completed:
            continue
        started = time.perf_counter()
        raw, usage = e3.oracle_base.call_chat_completions(
            args.base_url,
            args.api_key,
            args.model,
            e3.answer_prompt(
                row["question"], row["selected_context"], row["representation"]
            ),
            args.timeout,
            args.retries,
            args.answer_max_tokens,
        )
        row["run_signature"] = run_signature
        row["answer_latency_seconds"] = time.perf_counter() - started
        row["answer_attempted"] = True
        row["answer_raw_response"] = raw
        row["answer_usage"] = usage
        row["prediction"] = e3.oracle_base.extract_choice(raw, row["choices"])
        row["correct"] = e3.oracle_base.normalized_equal(row["prediction"], row["answer"])
        e3.append_jsonl(checkpoint_path, row)
        completed[key] = row
        print(f"Completed {number}/{len(rows)}", flush=True)

    question_order = {str(row["question_id"]): index for index, row in enumerate(questions)}
    final_rows = sorted(
        completed.values(),
        key=lambda row: (
            question_order[str(row["question_id"])],
            REPS.index(row["representation"]),
        ),
    )
    e3.write_jsonl(args.output_dir / "predictions.jsonl", final_rows)
    report = build_report(final_rows, configuration, answer_ran=True)
    report["validation"] = validation
    e3.write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
