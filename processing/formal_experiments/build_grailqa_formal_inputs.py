#!/usr/bin/env python3
"""Materialize and merge non-uniform GrailQA formal inputs by hop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
PROCESSING = PROJECT / "processing"
MINI_BUILDER = (
    PROCESSING / "grailqa_mini_kb_processing" / "build_mini_kb.py"
)
ALIAS_BUILDER = (
    PROCESSING / "grailqa_e0_processing" / "build_answer_alias_cache.py"
)
GRAIL_DATA = PROJECT / "GrailQA_v1.0 (1)" / "GrailQA_v1.0"
GRAIL_CODE = PROJECT / "GrailQA_v1.0 (1)" / "GrailQA"
DEFAULT_DATA = GRAIL_DATA / "grailqa_v1.0_dev.json"
DEFAULT_DATASETS = [
    GRAIL_DATA / "grailqa_v1.0_train.json",
    GRAIL_DATA / "grailqa_v1.0_dev.json",
]
DEFAULT_ENTITY_LINKING = (
    GRAIL_CODE / "entity_linking_results" / "grailqa_el.json"
)
DEFAULT_OUTPUT = HERE / "output" / "inputs" / "grailqa_100_100_60"
DEFAULT_PILOT = (
    PROCESSING
    / "grailqa_mini_kb_processing"
    / "output"
    / "dev_10_per_hop"
    / "selected_questions.jsonl"
)


def parse_counts(values: list[str]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for value in values:
        try:
            hop_text, count_text = value.split("=", 1)
            hop, count = int(hop_text), int(count_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid count {value!r}; expected HOP=COUNT"
            ) from exc
        if hop not in {1, 2, 3} or count < 1:
            raise argparse.ArgumentTypeError(f"Invalid formal count: {value!r}")
        counts[hop] = count
    if set(counts) != {1, 2, 3}:
        raise argparse.ArgumentTypeError("Counts must specify exactly hops 1, 2, and 3")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--endpoint",
        default=os.getenv("GRAILQA_SPARQL_ENDPOINT", "http://localhost:3001/sparql"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--counts",
        nargs="+",
        default=["1=100", "2=100", "3=60"],
        metavar="HOP=COUNT",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-questions",
        type=Path,
        default=DEFAULT_PILOT,
        help="Previously used pilot questions excluded from the formal sample.",
    )
    parser.add_argument(
        "--include-pilot",
        action="store_true",
        help="Disable the default pilot exclusion (not recommended).",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--max-consecutive-endpoint-failures", type=int, default=3)
    parser.add_argument("--label-batch-size", type=int, default=100)
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate candidate pools and write queries without contacting SPARQL.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge already completed per-hop part directories.",
    )
    parser.add_argument("--skip-alias-cache", action="store_true")
    parser.add_argument(
        "--datasets", type=Path, nargs="+", default=DEFAULT_DATASETS
    )
    parser.add_argument("--entity-linking", type=Path, default=DEFAULT_ENTITY_LINKING)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preferred_label(*values: Any) -> str | None:
    """Choose a stable, reasonably descriptive label across question-local aliases."""
    labels = {str(value).strip() for value in values if str(value or "").strip()}
    if not labels:
        return None
    return max(
        labels,
        key=lambda value: (
            len(value.split()),
            len(value),
            sum(character.isupper() for character in value),
            value.casefold(),
            value,
        ),
    )


def statement_semantics(value: Any) -> Any:
    """Remove presentation/provenance fields before comparing statement identity."""
    ignored = {
        "label",
        "predicate_label",
        "document",
        "triple",
        "graph",
        "source_question_ids",
        "source_hops",
    }
    if isinstance(value, dict):
        return {
            key: statement_semantics(item)
            for key, item in value.items()
            if key not in ignored
        }
    if isinstance(value, list):
        return [statement_semantics(item) for item in value]
    return value


def render_statement(row: dict[str, Any]) -> None:
    subject = row["subject"]
    obj = row["object"]
    subject_text = subject.get("label") or subject["id"]
    object_text = obj.get("label") or obj["id"]
    relation_id = row["predicate"]
    relation_text = row.get("predicate_label") or relation_id
    row["document"] = (
        f"{subject_text} has relation {relation_text} with {object_text}."
    )
    row["triple"] = f"({subject_text}, {relation_text}, {object_text})"
    row["graph"] = f"{subject_text} --[{relation_id}]--> {object_text}"


def run_parts(args: argparse.Namespace, counts: dict[int, int]) -> None:
    parts = args.output_dir / "_parts"
    for hop in (1, 2, 3):
        output = parts / f"hop_{hop}"
        command = [
            sys.executable,
            str(MINI_BUILDER),
            "--data",
            str(args.materialization_data),
            "--endpoint",
            args.endpoint,
            "--output-dir",
            str(output),
            "--hops",
            str(hop),
            "--questions-per-hop",
            str(counts[hop]),
            "--seed",
            str(args.seed),
            "--timeout",
            str(args.timeout),
            "--retries",
            str(args.retries),
            "--retry-backoff",
            str(args.retry_backoff),
            "--max-consecutive-endpoint-failures",
            str(args.max_consecutive_endpoint_failures),
            "--label-batch-size",
            str(args.label_batch_size),
        ]
        if args.skip_labels:
            command.append("--skip-labels")
        if args.no_resume:
            command.append("--no-resume")
        if args.dry_run:
            command.append("--dry-run")
        print("RUN", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=PROJECT, check=True)


def merge_statements(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        statement_id = str(row["statement_id"])
        current = merged.get(statement_id)
        if current is None:
            merged[statement_id] = dict(row)
            continue
        if statement_semantics(current) != statement_semantics(row):
            raise RuntimeError(f"Conflicting duplicate statement {statement_id}")
        for node_name in ("subject", "object"):
            label = preferred_label(
                current[node_name].get("label"), row[node_name].get("label")
            )
            if label:
                current[node_name]["label"] = label
        predicate_label = preferred_label(
            current.get("predicate_label"), row.get("predicate_label")
        )
        if predicate_label:
            current["predicate_label"] = predicate_label
        current["source_question_ids"] = sorted(
            set(current.get("source_question_ids", []))
            | set(row.get("source_question_ids", []))
        )
        current["source_hops"] = sorted(
            set(current.get("source_hops", [])) | set(row.get("source_hops", []))
        )
    statements = [merged[key] for key in sorted(merged)]
    for statement in statements:
        render_statement(statement)
    return statements


def merge_parts(args: argparse.Namespace, counts: dict[int, int]) -> dict[str, Any]:
    parts = [args.output_dir / "_parts" / f"hop_{hop}" for hop in (1, 2, 3)]
    for part in parts:
        summary_path = part / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing completed part: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("complete"):
            raise RuntimeError(f"Part is not complete: {summary_path}")

    questions = [
        row for part in parts for row in read_jsonl(part / "selected_questions.jsonl")
    ]
    questions.sort(key=lambda row: (int(row["hop_count"]), int(row["dataset_index"])))
    actual = Counter(int(row["hop_count"]) for row in questions)
    if dict(actual) != counts:
        raise RuntimeError(f"Expected formal counts {counts}, found {dict(actual)}")
    question_ids = [str(row["question_id"]) for row in questions]
    if len(question_ids) != len(set(question_ids)):
        raise RuntimeError("Duplicate question IDs across GrailQA hop parts")

    statements = merge_statements(
        row for part in parts for row in read_jsonl(part / "mini_kb.jsonl")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "selected_questions.jsonl", questions)
    write_jsonl(args.output_dir / "mini_kb.jsonl", statements)
    for filename in (
        "candidate_manifest.jsonl",
        "materialization_queries.jsonl",
        "query_results.jsonl",
        "failures.jsonl",
        "label_failures.jsonl",
    ):
        write_jsonl(
            args.output_dir / filename,
            (row for part in parts for row in read_jsonl(part / filename)),
        )

    report = {
        "status": "complete",
        "dataset": "GrailQA",
        "sampling": "deterministic per-hop candidate pools with SPARQL materialization",
        "seed": args.seed,
        "endpoint": args.endpoint,
        "pilot_exclusion_source": (
            str(args.exclude_questions.resolve())
            if not args.include_pilot and args.exclude_questions.exists()
            else None
        ),
        "requested_counts": {str(hop): counts[hop] for hop in sorted(counts)},
        "materialized_questions": len(questions),
        "materialized_by_hop": {str(hop): actual[hop] for hop in sorted(actual)},
        "canonical_statement_count": len(statements),
        "unique_question_ids": True,
        "selected_questions_sha256": sha256_file(
            args.output_dir / "selected_questions.jsonl"
        ),
        "mini_kb_sha256": sha256_file(args.output_dir / "mini_kb.jsonl"),
        "query_results_sha256": sha256_file(
            args.output_dir / "query_results.jsonl"
        ),
    }
    write_json(args.output_dir / "build_report.json", report)
    return report


def build_alias_cache(args: argparse.Namespace) -> None:
    required = [*args.datasets, args.entity_linking]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    command = [
        sys.executable,
        str(ALIAS_BUILDER),
        "--questions",
        str(args.output_dir / "selected_questions.jsonl"),
        "--datasets",
        *[str(path) for path in args.datasets],
        "--entity-linking",
        str(args.entity_linking),
        "--output",
        str(args.output_dir / "answer_alias_cache.json"),
    ]
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def main() -> None:
    args = parse_args()
    counts = parse_counts(args.counts)
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    if args.dry_run and args.merge_only:
        raise ValueError("--dry-run and --merge-only cannot be combined")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    excluded_ids: set[str] = set()
    if not args.include_pilot and args.exclude_questions.exists():
        excluded_ids = {
            str(row["question_id"])
            for row in read_jsonl(args.exclude_questions)
        }
    if excluded_ids:
        data = json.loads(args.data.read_text(encoding="utf-8"))
        found: set[str] = set()
        for item in data:
            if str(item.get("qid")) in excluded_ids:
                # Preserve list position so dataset_index remains the original
                # GrailQA dev index, while making the row ineligible.
                item["num_edge"] = -1
                item["_formal_excluded_pilot"] = True
                found.add(str(item["qid"]))
        if found != excluded_ids:
            missing = sorted(excluded_ids - found)
            raise RuntimeError(f"Pilot QIDs missing from source dev set: {missing}")
        materialization_data = args.output_dir / "_formal_source_dev.json"
        materialization_data.write_text(
            json.dumps(data, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        args.materialization_data = materialization_data
    else:
        args.materialization_data = args.data

    if not args.merge_only:
        run_parts(args, counts)
    if args.dry_run:
        report = {
            "status": "dry_run_complete",
            "dataset": "GrailQA",
            "requested_counts": {str(hop): counts[hop] for hop in sorted(counts)},
            "excluded_pilot_questions": len(excluded_ids),
            "pilot_exclusion_source": (
                str(args.exclude_questions.resolve()) if excluded_ids else None
            ),
            "parts": [
                str((args.output_dir / "_parts" / f"hop_{hop}").resolve())
                for hop in (1, 2, 3)
            ],
            "next_step": "rerun without --dry-run to contact the SPARQL endpoint",
        }
        write_json(args.output_dir / "dry_run_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    report = merge_parts(args, counts)
    if not args.skip_alias_cache:
        build_alias_cache(args)
        alias_path = args.output_dir / "answer_alias_cache.json"
        report["answer_alias_cache"] = str(alias_path.resolve())
        report["answer_alias_cache_sha256"] = sha256_file(alias_path)
        write_json(args.output_dir / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
