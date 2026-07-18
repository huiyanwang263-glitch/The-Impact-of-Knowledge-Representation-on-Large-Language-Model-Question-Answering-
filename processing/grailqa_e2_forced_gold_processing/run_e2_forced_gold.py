#!/usr/bin/env python3
"""Run GrailQA E2: E3-aligned noisy selection with forced Gold completion."""

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
E3_DIR = PROCESSING / "grailqa_e3_manual_llm_retrieval_processing"
sys.path.insert(0, str(E3_DIR))
import run_e3_manual_retrieval as e3  # noqa: E402


def force_complete_gold(
    selected_ids: list[str], gold_ids: list[str]
) -> tuple[list[str], dict[str, Any]]:
    """Insert all Gold while preserving the selector's context size when possible."""
    gold_set = set(gold_ids)
    selected_gold = [statement_id for statement_id in selected_ids if statement_id in gold_set]
    selected_noise = [statement_id for statement_id in selected_ids if statement_id not in gold_set]
    missing_gold = [statement_id for statement_id in gold_ids if statement_id not in selected_gold]
    target_count = max(len(selected_ids), len(gold_ids))
    keep_noise_count = target_count - len(gold_ids)
    kept_noise = selected_noise[:keep_noise_count]
    kept_noise_set = set(kept_noise)

    repaired = [
        statement_id
        for statement_id in selected_ids
        if statement_id in gold_set or statement_id in kept_noise_set
    ]
    repaired.extend(missing_gold)
    removed_noise = selected_noise[keep_noise_count:]

    if len(repaired) != target_count or not gold_set <= set(repaired):
        raise RuntimeError("Forced-Gold completion invariant failed")
    return repaired, {
        "raw_selected_statement_count": len(selected_ids),
        "forced_context_statement_count": len(repaired),
        "gold_added_statement_ids": missing_gold,
        "gold_added_statement_count": len(missing_gold),
        "selected_noise_removed_statement_ids": removed_noise,
        "selected_noise_removed_statement_count": len(removed_noise),
        "selected_noise_retained_statement_ids": kept_noise,
        "selected_noise_retained_statement_count": len(kept_noise),
        "context_size_preserved": len(repaired) == len(selected_ids),
    }


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]], answer_ran: bool) -> dict[str, Any]:
    before_rows = [
        {**row, "retrieval_metrics": row["retrieval_metrics_before_forcing"]}
        for row in rows
    ]
    before = e3.summarize(before_rows, False)
    after = e3.summarize(rows, False)
    result: dict[str, Any] = {
        "questions": len(rows),
        "selector_before_forcing": before,
        "forced_context": {
            "mean_statement_recall": after["mean_statement_recall_at_k"],
            "complete_sufficient_gold_rate": after[
                "complete_sufficient_gold_recall_at_k"
            ],
            "mean_statement_count": after["mean_selected_statement_count"],
            "min_statement_count": after["min_selected_statement_count"],
            "max_statement_count": after["max_selected_statement_count"],
            "mean_gold_statements_added": average(
                [float(row["forcing"]["gold_added_statement_count"]) for row in rows]
            ),
            "mean_selected_noise_statements_removed": average(
                [
                    float(row["forcing"]["selected_noise_removed_statement_count"])
                    for row in rows
                ]
            ),
            "context_size_preserved_rate": average(
                [float(row["forcing"]["context_size_preserved"]) for row in rows]
            ),
            "invalid_selector_fallback_rate": average(
                [
                    float(row["forcing"]["invalid_selector_fallback_used"])
                    for row in rows
                ]
            ),
        },
    }
    if answer_ran:
        result["answer"] = e3.e0.summarize(rows)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=e3.DEFAULT_QUESTIONS)
    parser.add_argument("--kb", type=Path, default=e3.DEFAULT_KB)
    parser.add_argument("--query-results", type=Path, default=e3.DEFAULT_QUERY_RESULTS)
    parser.add_argument("--datasets", type=Path, nargs="+", default=e3.DEFAULT_DATASETS)
    parser.add_argument("--alias-cache", type=Path, default=e3.DEFAULT_ALIAS_CACHE)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "output" / "e2_forced_gold_30"
    )
    parser.add_argument("--noise-statements", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-hop", type=int)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument(
        "--e3-selection-results",
        type=Path,
        help=(
            "Saved E3 selection_results.jsonl to reuse. Defaults to the complete "
            "E3 run; pass the E3 smoke path when using --max-per-hop 1."
        ),
    )
    parser.add_argument(
        "--run-selector",
        action="store_true",
        help="Run a fresh selector instead of reusing E3 (diagnostic only).",
    )
    parser.add_argument(
        "--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3-4b-16k"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--selector-max-tokens", type=int, default=1024)
    parser.add_argument("--answer-max-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        args.questions,
        args.kb,
        args.query_results,
        args.alias_cache,
        *args.datasets,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.noise_statements < 1 or args.top_k < 1:
        raise ValueError("--noise-statements and --top-k must be positive")
    if args.max_per_hop is not None and args.max_per_hop < 1:
        raise ValueError("--max-per-hop must be positive")

    all_questions = e3.read_jsonl(args.questions)
    questions = e3.select_questions(all_questions, args.max_per_hop)
    kb_rows = e3.read_jsonl(args.kb)
    kb_by_id = {row["statement_id"]: row for row in kb_rows}
    if len(kb_by_id) != len(kb_rows):
        raise RuntimeError("mini-KB contains duplicate statement IDs")

    evaluation_qids = {str(row["question_id"]) for row in all_questions}
    evaluation_gold_ids = {
        statement_id
        for row in all_questions
        for statement_id in row["gold_statement_ids"]
    }
    noise_rows, noise_audit = e3.offline_noise_bank(
        args.datasets, evaluation_qids, evaluation_gold_ids
    )
    query_results = e3.latest_successful_bindings(args.query_results)
    banks = e3.candidate_banks(
        questions,
        kb_by_id,
        query_results,
        noise_rows,
        args.noise_statements,
        args.seed,
    )
    used_noise_ids = {
        statement_id for bank in banks for statement_id in bank["noise_statement_ids"]
    }
    used_noise_rows = [
        row for row in noise_rows if row["statement_id"] in used_noise_ids
    ]
    alias_cache, alias_metadata = e3.e0.load_alias_cache(args.alias_cache)
    missing_alias_ids = sorted(
        {mid for bank in banks for mid in bank["gold_answer_ids"]} - set(alias_cache)
    )
    if missing_alias_ids:
        raise RuntimeError(f"Alias cache misses Gold MIDs: {missing_alias_ids[:20]}")

    max_gold = max(len(bank["sufficient_gold_statement_ids"]) for bank in banks)
    if max_gold > args.top_k:
        raise RuntimeError(
            f"Top-K {args.top_k} is smaller than maximum sufficient Gold size {max_gold}"
        )
    candidate_hash = e3.sha256_value(
        [
            {
                "question_id": bank["question_id"],
                "gold": bank["sufficient_gold_statement_ids"],
                "candidates": bank["candidate_statement_ids"],
            }
            for bank in banks
        ]
    )
    validation = {
        "condition": "E2_grailqa_noisy_selection_forced_gold",
        "questions": len(banks),
        "by_hop": {
            str(hop): sum(bank["hop_count"] == hop for bank in banks)
            for hop in sorted({bank["hop_count"] for bank in banks})
        },
        "representations": list(e3.REPRESENTATIONS),
        "full_freebase_retrieval_used": False,
        "noise_source": (
            "offline-grounded Gold edges from non-evaluation GrailQA train/dev questions"
        ),
        "noise_audit": noise_audit,
        "noise_statements_per_question": args.noise_statements,
        "deduplicated_background_noise_pool_count": len(noise_rows),
        "persisted_sampled_noise_union_count": len(used_noise_rows),
        "candidate_statement_count_min": min(
            len(bank["candidate_statement_ids"]) for bank in banks
        ),
        "candidate_statement_count_max": max(
            len(bank["candidate_statement_ids"]) for bank in banks
        ),
        "sufficient_gold_statement_count_min": min(
            len(bank["sufficient_gold_statement_ids"]) for bank in banks
        ),
        "sufficient_gold_statement_count_max": max_gold,
        "selector_top_k": args.top_k,
        "gold_present_in_every_candidate_bank": all(
            set(bank["sufficient_gold_statement_ids"])
            <= set(bank["candidate_statement_ids"])
            for bank in banks
        ),
        "gold_noise_disjoint": all(
            set(bank["sufficient_gold_statement_ids"]).isdisjoint(
                bank["noise_statement_ids"]
            )
            for bank in banks
        ),
        "candidate_ids_unique": all(
            len(bank["candidate_statement_ids"])
            == len(set(bank["candidate_statement_ids"]))
            for bank in banks
        ),
        "candidate_ids_and_order_shared_across_representations": True,
        "gold_forced_after_selector": True,
        "forced_context_size_rule": "max(raw_selected_count, sufficient_gold_count)",
        "invalid_selector_fallback": "sufficient_gold_only",
        "selector_policy": (
            "fresh_diagnostic_call" if args.run_selector else "reuse_saved_e3_selection"
        ),
        "candidate_bundle_sha256": candidate_hash,
        "alias_cache": alias_metadata,
        "alias_cache_loaded_into_prompt": False,
        "missing_alias_gold_mids": missing_alias_ids,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e3.write_jsonl(args.output_dir / "question_derived_noise_bank.jsonl", used_noise_rows)
    e3.write_jsonl(args.output_dir / "candidate_banks.jsonl", banks)
    e3.write_json(args.output_dir / "bundle_validation.json", validation)
    if args.build_only:
        report = {"status": "build_only", **validation}
        e3.write_json(args.output_dir / "metrics.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    e3_selection_path = args.e3_selection_results
    if e3_selection_path is None:
        e3_selection_path = E3_DIR / "output" / "e3_manual_30" / "selection_results.jsonl"
    e3_selections: dict[tuple[str, str], dict[str, Any]] = {}
    e3_selection_hash: str | None = None
    if not args.run_selector:
        if not e3_selection_path.exists():
            raise FileNotFoundError(
                f"Saved E3 selections not found: {e3_selection_path}. "
                "Pass --e3-selection-results or use --run-selector for diagnostics."
            )
        e3_metrics_path = e3_selection_path.parent / "metrics.json"
        if not e3_metrics_path.exists():
            raise FileNotFoundError(f"E3 metrics not found: {e3_metrics_path}")
        e3_metrics = e3.read_json(e3_metrics_path)
        e3_candidate_hash = e3_metrics.get("configuration", {}).get(
            "candidate_bundle_sha256"
        )
        if e3_candidate_hash != candidate_hash:
            raise RuntimeError(
                "E2/E3 candidate bundle mismatch: "
                f"E2={candidate_hash}, E3={e3_candidate_hash}"
            )
        e3_top_k = e3_metrics.get("configuration", {}).get("top_k")
        if e3_top_k != args.top_k:
            raise RuntimeError(f"E2/E3 Top-K mismatch: E2={args.top_k}, E3={e3_top_k}")
        reused_rows = e3.read_jsonl(e3_selection_path)
        for row in reused_rows:
            key = (str(row["question_id"]), str(row["representation"]))
            e3_selections[key] = row
        expected_keys = {
            (str(bank["question_id"]), representation)
            for bank in banks
            for representation in e3.REPRESENTATIONS
        }
        missing_keys = sorted(expected_keys - set(e3_selections))
        if missing_keys:
            raise RuntimeError(f"E3 selections miss cases: {missing_keys[:10]}")
        e3_selection_hash = e3.sha256_value(
            [
                {
                    "question_id": qid,
                    "representation": representation,
                    "selected_statement_ids": e3_selections[(qid, representation)][
                        "selected_statement_ids"
                    ],
                    "selector_valid": e3_selections[(qid, representation)][
                        "selector_valid"
                    ],
                }
                for qid, representation in sorted(expected_keys)
            ]
        )

    run_signature = e3.sha256_value(
        {
            "condition": validation["condition"],
            "candidate_bundle_sha256": candidate_hash,
            "model": args.model,
            "top_k": args.top_k,
            "max_per_hop": args.max_per_hop,
            "selector_max_tokens": args.selector_max_tokens,
            "answer_max_tokens": args.answer_max_tokens,
            "selection_only": args.selection_only,
            "forced_context_size_rule": validation["forced_context_size_rule"],
            "invalid_selector_fallback": validation["invalid_selector_fallback"],
            "selector_policy": validation["selector_policy"],
            "e3_selection_sha256": e3_selection_hash,
        }
    )
    checkpoint_path = args.output_dir / "e2_checkpoint.jsonl"
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    for row in e3.read_jsonl(checkpoint_path):
        if row.get("run_signature") == run_signature:
            cached[(str(row["question_id"]), str(row["representation"]))] = row

    results: list[dict[str, Any]] = []
    total = len(banks) * len(e3.REPRESENTATIONS)
    completed = 0
    for bank in banks:
        bank_by_id = {
            statement["statement_id"]: statement
            for statement in bank["candidate_statements"]
        }
        for representation in e3.REPRESENTATIONS:
            completed += 1
            cache_key = (bank["question_id"], representation)
            if cache_key in cached:
                results.append(cached[cache_key])
                continue

            if args.run_selector:
                started = time.perf_counter()
                raw_selector, selector_usage = e3.call_selector(
                    args.base_url,
                    args.model,
                    e3.selector_prompt(
                        bank["question"],
                        e3.render_candidates(bank, representation),
                        args.top_k,
                    ),
                    args.top_k,
                    args.timeout,
                    args.retries,
                    args.selector_max_tokens,
                )
                selector_latency = time.perf_counter() - started
                raw_selected_ids, unknown_ids, selector_valid, selector_error = (
                    e3.parse_selection(
                        raw_selector,
                        set(bank["candidate_statement_ids"]),
                        args.top_k,
                    )
                )
                selector_source = "fresh_diagnostic_call"
            else:
                source = e3_selections[cache_key]
                raw_selector = str(source.get("raw_selector_response", ""))
                selector_usage = dict(source.get("selector_usage", {}))
                selector_latency = float(source.get("selector_latency_seconds", 0.0))
                raw_selected_ids = list(source.get("selected_statement_ids", []))
                unknown_ids = list(source.get("unknown_statement_ids", []))
                selector_valid = bool(source.get("selector_valid"))
                selector_error = source.get("selector_error")
                selector_source = "saved_e3_selection"
                valid_candidate_ids = set(bank["candidate_statement_ids"])
                if any(
                    statement_id not in valid_candidate_ids
                    for statement_id in raw_selected_ids
                ):
                    raise RuntimeError(f"E3 selection contains invalid IDs for {cache_key}")
            forced_ids, forcing = force_complete_gold(
                raw_selected_ids, bank["sufficient_gold_statement_ids"]
            )
            forcing["invalid_selector_fallback_used"] = not selector_valid

            row: dict[str, Any] = {
                "run_signature": run_signature,
                "question_id": bank["question_id"],
                "dataset_index": bank["dataset_index"],
                "hop_count": bank["hop_count"],
                "level": bank["level"],
                "representation": representation,
                "question": bank["question"],
                "gold_answers": bank["gold_answers"],
                "gold_answer_ids": bank["gold_answer_ids"],
                "sufficient_gold_statement_ids": bank[
                    "sufficient_gold_statement_ids"
                ],
                "raw_selected_statement_ids": raw_selected_ids,
                "selected_statement_ids": forced_ids,
                "forcing": forcing,
                "unknown_statement_ids": unknown_ids,
                "selector_valid": selector_valid,
                "selector_error": selector_error,
                "selector_source": selector_source,
                "raw_selector_response": raw_selector,
                "selector_usage": selector_usage,
                "selector_latency_seconds": selector_latency,
                "retrieval_metrics_before_forcing": e3.retrieval_metrics(
                    raw_selected_ids, bank["sufficient_gold_statement_ids"]
                ),
                "retrieval_metrics": e3.retrieval_metrics(
                    forced_ids, bank["sufficient_gold_statement_ids"]
                ),
                "answer_attempted": False,
                "prediction": [],
                "raw_response": "",
                "output_valid": False,
                "parse_error": None,
                **e3.e0.score_canonical_answer_set([], bank["gold_answers"]),
                **e3.e0.score_alias_answer_set(
                    [], bank["gold_answer_ids"], alias_cache
                ),
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "latency_seconds": 0.0,
            }
            if not args.selection_only:
                context = e3.selected_context(forced_ids, bank_by_id, representation)
                answer_started = time.perf_counter()
                raw_answer, answer_usage = e3.e0.call_ollama(
                    args.base_url,
                    args.model,
                    e3.answer_prompt(bank["question"], context, representation),
                    args.timeout,
                    args.retries,
                    args.answer_max_tokens,
                )
                answer_latency = time.perf_counter() - answer_started
                predicted, valid, parse_error = e3.e0.parse_answers(raw_answer)
                row.update(
                    {
                        "answer_attempted": True,
                        "prediction": predicted,
                        "raw_response": raw_answer,
                        "output_valid": valid,
                        "parse_error": parse_error,
                        **e3.e0.score_canonical_answer_set(
                            predicted, bank["gold_answers"]
                        ),
                        **e3.e0.score_alias_answer_set(
                            predicted, bank["gold_answer_ids"], alias_cache
                        ),
                        "usage": answer_usage,
                        "latency_seconds": answer_latency,
                    }
                )
            e3.append_jsonl(checkpoint_path, row)
            results.append(row)
            print(
                f"Completed {completed}/{total} "
                f"({bank['hop_count']}-hop, {representation})",
                flush=True,
            )

    results.sort(
        key=lambda row: (
            row["hop_count"],
            row["dataset_index"],
            e3.REPRESENTATIONS.index(row["representation"]),
        )
    )
    e3.write_jsonl(args.output_dir / "selection_results.jsonl", results)
    if not args.selection_only:
        e3.write_jsonl(args.output_dir / "predictions.jsonl", results)
    report = {
        "status": "completed",
        "condition": validation["condition"],
        "model": args.model,
        "questions": len(banks),
        "cases": len(results),
        "selector_cases": len(results),
        "selector_calls": len(results) if args.run_selector else 0,
        "new_selector_calls": len(results) if args.run_selector else 0,
        "reused_e3_selections": 0 if args.run_selector else len(results),
        "answer_calls": 0
        if args.selection_only
        else sum(bool(row["answer_attempted"]) for row in results),
        "selection_only": args.selection_only,
        "configuration": {
            "noise_statements_per_question": args.noise_statements,
            "top_k": args.top_k,
            "gold_forced_after_selector": True,
            "forced_context_size_rule": validation["forced_context_size_rule"],
            "invalid_selector_fallback": validation["invalid_selector_fallback"],
            "selector_policy": validation["selector_policy"],
            "e3_selection_results": None
            if args.run_selector
            else str(e3_selection_path.resolve()),
            "e3_selection_sha256": e3_selection_hash,
            "temperature": 0,
            "think": False,
            "candidate_bundle_sha256": candidate_hash,
            "alias_cache": alias_metadata,
            "alias_cache_loaded_into_prompt": False,
        },
        "by_representation": {
            representation: summarize(
                [row for row in results if row["representation"] == representation],
                not args.selection_only,
            )
            for representation in e3.REPRESENTATIONS
        },
        "by_hop": {
            str(hop): {
                "by_representation": {
                    representation: summarize(
                        [
                            row
                            for row in results
                            if row["hop_count"] == hop
                            and row["representation"] == representation
                        ],
                        not args.selection_only,
                    )
                    for representation in e3.REPRESENTATIONS
                }
            }
            for hop in sorted({row["hop_count"] for row in results})
        },
    }
    e3.write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
