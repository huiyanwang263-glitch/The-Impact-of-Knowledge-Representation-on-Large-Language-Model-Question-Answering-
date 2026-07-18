#!/usr/bin/env python3
"""Build a tiny grounded GrailQA KB with 10 successful questions per hop."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import urllib.error
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DEFAULT_DATA = (
    PROJECT
    / "GrailQA_v1.0 (1)"
    / "GrailQA_v1.0"
    / "grailqa_v1.0_dev.json"
)
HELPER_PATH = HERE.parent / "grailqa_gold_path_processing" / "build_gold_paths.py"
DEFAULT_ENDPOINT = os.environ.get(
    "GRAILQA_SPARQL_ENDPOINT",
    "http://cse-cnc238814s.cse.ohio-state.edu:3093/sparql",
)


def load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("grailqa_gold_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper module from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def eligible(item: dict[str, Any], hop: int) -> bool:
    return (
        int(item.get("num_edge", -1)) == hop
        and item.get("function") == "none"
        and bool(item.get("answer"))
        and all(answer.get("answer_type") == "Entity" for answer in item["answer"])
    )


def build_candidate_pools(
    data: list[dict[str, Any]], hops: list[int], seed: int
) -> dict[int, list[tuple[int, dict[str, Any]]]]:
    pools: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for hop in hops:
        rows = [(index, item) for index, item in enumerate(data) if eligible(item, hop)]
        random.Random(seed + hop).shuffle(rows)
        pools[hop] = rows
    return pools


def load_latest_checkpoints(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[str(row["question_id"])] = row
    return latest


def serialize_statement(statement: dict[str, Any]) -> dict[str, Any]:
    subject = statement["subject"]
    obj = statement["object"]
    subject_text = subject.get("label") or subject["id"]
    object_text = obj.get("label") or obj["id"]
    relation_id = statement["predicate"]
    relation_text = statement.get("predicate_label") or relation_id
    return {
        **statement,
        "document": f"{subject_text} has relation {relation_text} with {object_text}.",
        "triple": f"({subject_text}, {relation_text}, {object_text})",
        "graph": f"{subject_text} --[{relation_id}]--> {object_text}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "output" / "dev_10_per_hop"
    )
    parser.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--questions-per-hop", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--max-consecutive-endpoint-failures", type=int, default=3)
    parser.add_argument("--label-batch-size", type=int, default=100)
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helper = load_helper()
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    if not args.hops or any(hop not in {1, 2, 3} for hop in args.hops):
        raise ValueError("--hops must contain only 1, 2, and/or 3")
    if args.questions_per_hop < 1:
        raise ValueError("--questions-per-hop must be positive")

    hops = sorted(set(args.hops))
    data = json.loads(args.data.read_text(encoding="utf-8"))
    pools = build_candidate_pools(data, hops, args.seed)
    if any(len(pools[hop]) < args.questions_per_hop for hop in hops):
        sizes = {str(hop): len(pools[hop]) for hop in hops}
        raise ValueError(f"Insufficient eligible candidates: {sizes}")

    manifest = [
        {
            "hop_count": hop,
            "candidate_rank": rank,
            "dataset_index": index,
            "question_id": str(item["qid"]),
            "question": item["question"],
            "level": item.get("level"),
        }
        for hop in hops
        for rank, (index, item) in enumerate(pools[hop], start=1)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "candidate_manifest.jsonl", manifest)

    prepared: dict[str, dict[str, Any]] = {}
    preparation_failures: list[dict[str, Any]] = []
    for hop in hops:
        for rank, (dataset_index, item) in enumerate(pools[hop], start=1):
            question_id = str(item["qid"])
            try:
                query, variables = helper.materialization_query(item)
                prepared[question_id] = {
                    "hop_count": hop,
                    "candidate_rank": rank,
                    "dataset_index": dataset_index,
                    "item": item,
                    "query": query,
                    "variables": variables,
                }
            except Exception as exc:
                preparation_failures.append(
                    {
                        "status": "failure",
                        "question_id": question_id,
                        "hop_count": hop,
                        "candidate_rank": rank,
                        "dataset_index": dataset_index,
                        "stage": "query_rewrite",
                        "error": repr(exc),
                    }
                )

    write_jsonl(
        args.output_dir / "materialization_queries.jsonl",
        (
            {
                "question_id": question_id,
                "hop_count": row["hop_count"],
                "candidate_rank": row["candidate_rank"],
                "dataset_index": row["dataset_index"],
                "question": row["item"]["question"],
                "variables": row["variables"],
                "query": row["query"],
            }
            for question_id, row in sorted(
                prepared.items(),
                key=lambda pair: (pair[1]["hop_count"], pair[1]["candidate_rank"]),
            )
        ),
    )

    base_summary = {
        "data": str(args.data.resolve()),
        "requested_hops": hops,
        "questions_per_hop": args.questions_per_hop,
        "seed": args.seed,
        "eligible_candidates_by_hop": {
            str(hop): len(pools[hop]) for hop in hops
        },
        "queries_prepared": len(prepared),
        "query_rewrite_failures": len(preparation_failures),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        write_jsonl(args.output_dir / "failures.jsonl", preparation_failures)
        write_json(args.output_dir / "summary.json", base_summary)
        print(json.dumps(base_summary, ensure_ascii=False, indent=2))
        return

    checkpoint_path = args.output_dir / "query_results.jsonl"
    if args.no_resume and checkpoint_path.exists():
        raise RuntimeError(
            f"{checkpoint_path} already exists; use a new output directory"
        )
    latest = {} if args.no_resume else load_latest_checkpoints(checkpoint_path)
    latest_failures = {str(row["question_id"]): row for row in preparation_failures}
    selected_ids: dict[int, list[str]] = {hop: [] for hop in hops}
    consecutive_endpoint_failures = 0
    endpoint_abort_reason: str | None = None

    for hop in hops:
        for candidate_rank, (dataset_index, item) in enumerate(pools[hop], start=1):
            if len(selected_ids[hop]) >= args.questions_per_hop:
                break
            question_id = str(item["qid"])
            prep = prepared.get(question_id)
            if prep is None:
                continue
            query_hash = helper.stable_id("query", prep["query"])
            cached = latest.get(question_id)
            if (
                cached
                and cached.get("status") == "success"
                and cached.get("query_hash") == query_hash
                and cached.get("endpoint") == args.endpoint
            ):
                selected_ids[hop].append(question_id)
                latest_failures.pop(question_id, None)
                continue
            try:
                response = helper.query_with_retries(
                    args.endpoint,
                    prep["query"],
                    args.timeout,
                    args.retries,
                    args.retry_backoff,
                )
                bindings, incomplete = helper.parse_bindings(response, prep["variables"])
                valid, expected, observed = helper.validate_denotation(item, bindings)
                if not bindings:
                    raise ValueError("SPARQL returned no complete node bindings")
                if not valid:
                    raise ValueError(
                        "Denotation mismatch: "
                        f"expected={sorted(expected)!r}, observed={sorted(observed)!r}"
                    )
                row = {
                    "status": "success",
                    "question_id": question_id,
                    "hop_count": hop,
                    "candidate_rank": candidate_rank,
                    "dataset_index": dataset_index,
                    "query_hash": query_hash,
                    "endpoint": args.endpoint,
                    "variables": prep["variables"],
                    "bindings": bindings,
                    "incomplete_bindings": incomplete,
                }
                append_jsonl(checkpoint_path, row)
                latest[question_id] = row
                selected_ids[hop].append(question_id)
                latest_failures.pop(question_id, None)
                consecutive_endpoint_failures = 0
                print(
                    f"hop={hop}: materialized {len(selected_ids[hop])}/"
                    f"{args.questions_per_hop} (candidate rank {candidate_rank})",
                    flush=True,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                consecutive_endpoint_failures += 1
                row = {
                    "status": "failure",
                    "question_id": question_id,
                    "hop_count": hop,
                    "candidate_rank": candidate_rank,
                    "dataset_index": dataset_index,
                    "query_hash": query_hash,
                    "endpoint": args.endpoint,
                    "stage": "endpoint",
                    "error": repr(exc),
                }
                append_jsonl(checkpoint_path, row)
                latest[question_id] = row
                latest_failures[question_id] = row
                if consecutive_endpoint_failures >= args.max_consecutive_endpoint_failures:
                    endpoint_abort_reason = (
                        f"Aborted after {consecutive_endpoint_failures} consecutive "
                        f"endpoint failures; last error: {exc!r}"
                    )
                    break
            except Exception as exc:
                consecutive_endpoint_failures = 0
                row = {
                    "status": "failure",
                    "question_id": question_id,
                    "hop_count": hop,
                    "candidate_rank": candidate_rank,
                    "dataset_index": dataset_index,
                    "query_hash": query_hash,
                    "endpoint": args.endpoint,
                    "stage": "sparql_or_validation",
                    "error": repr(exc),
                }
                append_jsonl(checkpoint_path, row)
                latest[question_id] = row
                latest_failures[question_id] = row
        if endpoint_abort_reason:
            break

    labels: dict[str, str] = {}
    entity_ids: set[str] = set()
    for hop in hops:
        for question_id in selected_ids[hop]:
            prep = prepared[question_id]
            labels.update(helper.initial_labels(prep["item"]))
            for binding in latest[question_id]["bindings"]:
                for node in binding.values():
                    if node["kind"] == "entity":
                        entity_ids.add(node["id"])

    label_failures: list[dict[str, Any]] = []
    if not args.skip_labels and entity_ids:
        fetched, label_failures = helper.fetch_labels(
            args.endpoint,
            entity_ids - set(labels),
            args.timeout,
            args.retries,
            args.retry_backoff,
            args.label_batch_size,
        )
        labels.update(fetched)

    questions: list[dict[str, Any]] = []
    canonical: dict[str, dict[str, Any]] = {}
    for hop in hops:
        for question_id in selected_ids[hop]:
            prep = prepared[question_id]
            result = latest[question_id]
            record = helper.build_question_record(
                prep["item"],
                prep["dataset_index"],
                prep["query"],
                result["bindings"],
                labels,
                result.get("incomplete_bindings", 0),
            )
            record["candidate_rank"] = prep["candidate_rank"]
            record["materialized_from_endpoint"] = args.endpoint
            questions.append(record)
            for statement in record["gold_statements"]:
                statement_id = statement["statement_id"]
                if statement_id not in canonical:
                    canonical[statement_id] = {
                        **serialize_statement(statement),
                        "source_question_ids": [],
                        "source_hops": [],
                    }
                if question_id not in canonical[statement_id]["source_question_ids"]:
                    canonical[statement_id]["source_question_ids"].append(question_id)
                if hop not in canonical[statement_id]["source_hops"]:
                    canonical[statement_id]["source_hops"].append(hop)

    questions.sort(key=lambda row: (row["hop_count"], row["dataset_index"]))
    statements = sorted(canonical.values(), key=lambda row: row["statement_id"])
    write_jsonl(args.output_dir / "selected_questions.jsonl", questions)
    write_jsonl(args.output_dir / "mini_kb.jsonl", statements)
    write_jsonl(
        args.output_dir / "failures.jsonl",
        sorted(
            latest_failures.values(),
            key=lambda row: (row.get("hop_count", -1), row.get("candidate_rank", -1)),
        ),
    )
    write_jsonl(args.output_dir / "label_failures.jsonl", label_failures)

    selected_counts = Counter(str(row["hop_count"]) for row in questions)
    complete = all(
        selected_counts[str(hop)] == args.questions_per_hop for hop in hops
    )
    summary = {
        **base_summary,
        "dry_run": False,
        "endpoint": args.endpoint,
        "complete": complete,
        "materialized_questions": len(questions),
        "materialized_by_hop": dict(selected_counts),
        "canonical_statement_count": len(statements),
        "current_failure_count": len(latest_failures),
        "endpoint_abort_reason": endpoint_abort_reason,
        "label_lookup_failure_batches": len(label_failures),
        "entities_without_friendly_label": len(entity_ids - set(labels)),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed query checkpoints are preserved.", file=sys.stderr)
        raise SystemExit(130)
