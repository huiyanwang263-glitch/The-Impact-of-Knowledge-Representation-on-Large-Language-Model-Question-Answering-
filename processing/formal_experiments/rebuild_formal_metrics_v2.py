#!/usr/bin/env python3
"""Re-score completed formal runs without making any model API calls."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
PROCESSING = HERE.parent
PROJECT = PROCESSING.parent
E0_DIR = PROCESSING / "grailqa_e0_processing"
E1_DIR = PROCESSING / "grailqa_e1_gold_only_processing"
E3_DIR = PROCESSING / "grailqa_e3_manual_llm_retrieval_processing"
E2_DIR = PROCESSING / "grailqa_e2_forced_gold_processing"
for module_dir in (E0_DIR, E1_DIR, E3_DIR, E2_DIR, HERE):
    sys.path.insert(0, str(module_dir))

import run_e0_closed_book as e0  # noqa: E402
import run_e1_gold_only as e1  # noqa: E402
import run_e3_manual_retrieval as e3  # noqa: E402
import run_e2_forced_gold as e2  # noqa: E402
from deepinfra_run_metadata import aggregate_usage_views  # noqa: E402


DEFAULT_MODEL_DIR = (
    "Qwen_Qwen3-14B_01b0452aeedc42eaf4a73d87654a8129d6a436f0"
)
ALIAS_RESULT_FIELDS = (
    "alias_scoring_version",
    "alias_scoring_unit",
    "alias_answer_set_em",
    "alias_answer_precision",
    "alias_answer_recall",
    "alias_answer_f1",
    "alias_matches",
    "alias_gold_group_count",
    "alias_gold_groups",
    "unmatched_predictions",
    "unmatched_gold_groups",
    "unmatched_gold_ids",
)
CANONICAL_RESULT_FIELDS = (
    "canonical_answer_set_em",
    "canonical_answer_precision",
    "canonical_answer_recall",
    "canonical_answer_f1",
    "normalized_prediction_set",
    "normalized_canonical_gold_set",
)
REPRESENTATIONS = ("document", "triple", "graph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=HERE / "output" / "runs",
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=HERE / "output" / "reports" / "deepinfra_qwen3_14b_v2",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def usage_annotated_metrics(stage_dir: Path, original: dict[str, Any]) -> dict[str, Any]:
    metrics = copy.deepcopy(original)
    usage_views = aggregate_usage_views(stage_dir)
    metrics["metrics_schema_version"] = 2
    metrics["deepinfra_usage"] = usage_views["incremental_billed"]
    metrics["deepinfra_usage_incremental"] = usage_views["incremental_billed"]
    metrics["deepinfra_usage_condition_equivalent"] = usage_views[
        "condition_equivalent"
    ]
    metrics["deepinfra_usage_accounting"] = {
        "schema_version": 2,
        "deepinfra_usage_alias": "incremental_billed",
        "incremental_billed_excludes_saved_e3_selector_usage": True,
        "condition_equivalent_includes_saved_e3_selector_usage": True,
    }
    return metrics


def rescore_alias_rows(
    rows: list[dict[str, Any]],
    alias_cache: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rescored: list[dict[str, Any]] = []
    for original in rows:
        row = copy.deepcopy(original)
        row["canonical_evaluation_v1"] = {
            key: row[key]
            for key in CANONICAL_RESULT_FIELDS
            if key in row
        }
        row["alias_evaluation_v1"] = {
            key: row[key]
            for key in ALIAS_RESULT_FIELDS
            if key in row
        }
        for key in ALIAS_RESULT_FIELDS:
            row.pop(key, None)
        row.update(
            e0.score_canonical_answer_set(
                row["prediction"],
                row["gold_answers"],
            )
        )
        row.update(
            e0.score_alias_answer_set(
                row["prediction"],
                row["gold_answers"],
                row["gold_answer_ids"],
                alias_cache,
            )
        )
        if row["canonical_answer_set_em"] and not row["alias_answer_set_em"]:
            raise RuntimeError(
                "Grouped alias scoring violated canonical dominance for "
                f"{row['question_id']} / {row.get('representation')}"
            )
        rescored.append(row)
    return rescored


def replace_grail_summaries(
    stage: str,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    hops = sorted({int(row["hop_count"]) for row in rows})
    if stage == "e0":
        metrics["overall"] = e0.summarize(rows)
        metrics["by_hop"] = {
            str(hop): e0.summarize(
                [row for row in rows if int(row["hop_count"]) == hop]
            )
            for hop in hops
        }
        return

    if stage == "e1":
        summarizer: Callable[[list[dict[str, Any]]], dict[str, Any]] = e0.summarize
    elif stage == "e3":
        answer_ran = not bool(metrics.get("selection_only"))
        summarizer = lambda selected: e3.summarize(selected, answer_ran)
    elif stage == "e2":
        answer_ran = not bool(metrics.get("selection_only"))
        summarizer = lambda selected: e2.summarize(selected, answer_ran)
    else:
        raise ValueError(stage)

    metrics["by_representation"] = {
        representation: summarizer(
            [row for row in rows if row["representation"] == representation]
        )
        for representation in REPRESENTATIONS
    }
    metrics["by_hop"] = {
        str(hop): {
            "by_representation": {
                representation: summarizer(
                    [
                        row
                        for row in rows
                        if int(row["hop_count"]) == hop
                        and row["representation"] == representation
                    ]
                )
                for representation in REPRESENTATIONS
            }
        }
        for hop in hops
    }


def row_key(stage: str, row: dict[str, Any]) -> tuple[str, ...]:
    if stage == "e0":
        return (str(row["question_id"]),)
    return (str(row["question_id"]), str(row["representation"]))


def assert_stage_rows(dataset: str, stage: str, rows: list[dict[str, Any]]) -> None:
    expected = 260 if stage == "e0" else 780
    keys = [row_key(stage, row) for row in rows]
    if len(rows) != expected or len(set(keys)) != expected:
        raise RuntimeError(
            f"{dataset}/{stage} expected {expected} unique rows; "
            f"found {len(rows)} rows and {len(set(keys))} unique keys"
        )


def validate_e2_reuse(dataset_root: Path) -> dict[str, Any]:
    e3_rows = read_jsonl(dataset_root / "e3" / "predictions.jsonl")
    e2_rows = read_jsonl(dataset_root / "e2" / "predictions.jsonl")
    e3_by_key = {row_key("e3", row): row for row in e3_rows}
    e2_by_key = {row_key("e2", row): row for row in e2_rows}
    same_keys = e3_by_key.keys() == e2_by_key.keys()
    reused_selections = sum(
        e2_by_key[key]["raw_selected_statement_ids"]
        == e3_by_key[key]["selected_statement_ids"]
        for key in e3_by_key
    )
    reused_response_ids = sum(
        (e2_by_key[key].get("selector_usage") or {}).get("response_id")
        == (e3_by_key[key].get("selector_usage") or {}).get("response_id")
        for key in e3_by_key
    )
    gold_field = (
        "gold_statement_ids"
        if dataset_root.parent.parent.name == "kqapro"
        else "sufficient_gold_statement_ids"
    )
    complete_gold = sum(
        set(row[gold_field]) <= set(row["selected_statement_ids"])
        for row in e2_rows
    )
    if not (
        same_keys
        and reused_selections == 780
        and reused_response_ids == 780
        and complete_gold == 780
    ):
        raise RuntimeError(f"E2 reuse validation failed for {dataset_root}")
    return {
        "same_case_keys": same_keys,
        "reused_selections": reused_selections,
        "reused_selector_response_ids": reused_response_ids,
        "complete_gold_after_forcing": complete_gold,
    }


def main() -> None:
    args = parse_args()
    audit: dict[str, Any] = {
        "status": "completed",
        "metrics_schema_version": 2,
        "model_dir": args.model_dir,
        "model_api_calls_made": 0,
        "datasets": {},
    }
    total_incremental_cost = 0.0
    total_condition_cost = 0.0

    grail_root = args.runs_root / "grailqa" / "deepinfra" / args.model_dir
    alias_path = (
        HERE
        / "output"
        / "inputs"
        / "grailqa_100_100_60"
        / "answer_alias_cache.json"
    )
    alias_cache, alias_metadata = e0.load_alias_cache(alias_path)

    for dataset in ("kqapro", "grailqa"):
        dataset_root = args.runs_root / dataset / "deepinfra" / args.model_dir
        dataset_audit: dict[str, Any] = {
            "stages": {},
            "e2_reuse": validate_e2_reuse(dataset_root),
        }
        for stage in ("e0", "e1", "e3", "e2"):
            stage_dir = dataset_root / stage
            predictions_path = stage_dir / "predictions.jsonl"
            metrics_path = stage_dir / "metrics.json"
            rows = read_jsonl(predictions_path)
            assert_stage_rows(dataset, stage, rows)
            original_metrics = read_json(metrics_path)
            if original_metrics.get("status") != "completed":
                raise RuntimeError(f"{dataset}/{stage} is not completed")

            metrics_v2 = usage_annotated_metrics(stage_dir, original_metrics)
            if dataset == "grailqa":
                rows_v2 = rescore_alias_rows(rows, alias_cache)
                write_jsonl(stage_dir / "predictions_v2.jsonl", rows_v2)
                replace_grail_summaries(stage, metrics_v2, rows_v2)
                metrics_v2["evaluation"] = {
                    "schema_version": 2,
                    "canonical_unit": "normalized_answer_string_set",
                    "alias_aware_unit": "normalized_gold_answer_group",
                    "gold_mids_with_same_normalized_answer_are_grouped": True,
                    "normalization_version": 2,
                    "single_token_articles_are_preserved": True,
                    "alias_cache": alias_metadata,
                    "model_api_calls_made": 0,
                }
            write_json(stage_dir / "metrics_v2.json", metrics_v2)

            incremental = metrics_v2["deepinfra_usage_incremental"]
            condition = metrics_v2["deepinfra_usage_condition_equivalent"]
            total_incremental_cost += float(incremental["estimated_cost_usd"])
            total_condition_cost += float(condition["estimated_cost_usd"])
            dataset_audit["stages"][stage] = {
                "row_count": len(rows),
                "unique_row_keys": len({row_key(stage, row) for row in rows}),
                "original_predictions_sha256": sha256_file(predictions_path),
                "original_metrics_sha256": sha256_file(metrics_path),
                "metrics_v2": str((stage_dir / "metrics_v2.json").resolve()),
                "predictions_v2": (
                    str((stage_dir / "predictions_v2.jsonl").resolve())
                    if dataset == "grailqa"
                    else None
                ),
                "incremental_usage": incremental,
                "condition_equivalent_usage": condition,
            }
        audit["datasets"][dataset] = dataset_audit

    audit["cost_summary_usd"] = {
        "incremental_billed_total": total_incremental_cost,
        "condition_equivalent_stage_sum": total_condition_cost,
        "reused_e3_selector_cost_in_e2": (
            total_condition_cost - total_incremental_cost
        ),
    }
    audit["alias_cache"] = {
        **alias_metadata,
        "path": str(alias_path.resolve()),
    }
    write_json(args.report_dir / "formal_audit_v2.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
