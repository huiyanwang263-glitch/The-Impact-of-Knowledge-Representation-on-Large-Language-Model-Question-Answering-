#!/usr/bin/env python3
"""Build and freeze a non-uniform KQA Pro formal question manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
ORACLE_DIR = PROJECT / "processing" / "knowledge_representation_processing"
sys.path.insert(0, str(ORACLE_DIR))
import run_oracle_hops_15 as oracle  # noqa: E402


DEFAULT_OUTPUT = HERE / "output" / "inputs" / "kqapro_100_100_60"
DEFAULT_PILOT = (
    PROJECT
    / "processing"
    / "knowledge_representation_processing"
    / "output"
    / "oracle_hops_15"
    / "oracle_questions.jsonl"
)
REPRESENTATIONS = ("document", "triple", "graph")


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
            raise argparse.ArgumentTypeError(
                f"Invalid count {value!r}; hops must be 1/2/3 and counts positive"
            )
        counts[hop] = count
    if set(counts) != {1, 2, 3}:
        raise argparse.ArgumentTypeError("Counts must specify exactly 1, 2, and 3 hops")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, default=PROJECT / "kb.json")
    parser.add_argument("--train", type=Path, default=PROJECT / "train.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--counts",
        nargs="+",
        default=["1=100", "2=100", "3=60"],
        metavar="HOP=COUNT",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=DEFAULT_PILOT,
        help="Previously used pilot questions excluded from the formal sample.",
    )
    parser.add_argument(
        "--include-pilot",
        action="store_true",
        help="Disable the default pilot exclusion (not recommended).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing frozen manifest in the target directory.",
    )
    return parser.parse_args()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(compact_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def validate(
    rows: list[dict[str, Any]],
    counts: dict[int, int],
    excluded_indices: set[int],
) -> dict[str, Any]:
    actual = Counter(int(row["hop_count"]) for row in rows)
    if dict(actual) != counts:
        raise RuntimeError(f"Expected hop counts {counts}, found {dict(actual)}")
    question_ids = [str(row["question_id"]) for row in rows]
    train_indices = [int(row["train_index"]) for row in rows]
    if len(question_ids) != len(set(question_ids)):
        raise RuntimeError("Duplicate question IDs in formal manifest")
    if len(train_indices) != len(set(train_indices)):
        raise RuntimeError("Duplicate train indices in formal manifest")
    overlap = sorted(set(train_indices) & excluded_indices)
    if overlap:
        raise RuntimeError(f"Formal manifest overlaps excluded pilot indices: {overlap}")
    for row in rows:
        hop = int(row["hop_count"])
        gold_ids = list(row["gold_statement_ids"])
        gold = list(row["gold_statements"])
        if len(gold_ids) != hop or len(gold) != hop:
            raise RuntimeError(
                f"{row['question_id']}: hop count and Gold path length disagree"
            )
        if gold_ids != [statement["statement_id"] for statement in gold]:
            raise RuntimeError(f"{row['question_id']}: Gold statement IDs are misaligned")
        if row["answer"] not in row["choices"]:
            raise RuntimeError(f"{row['question_id']}: Gold answer is not a choice")
        if set(row["contexts"]) != set(REPRESENTATIONS):
            raise RuntimeError(f"{row['question_id']}: representation contexts missing")
    return {
        "questions": len(rows),
        "questions_by_hop": {str(hop): actual[hop] for hop in sorted(actual)},
        "unique_question_ids": True,
        "unique_train_indices": True,
        "pilot_overlap_count": 0,
        "single_relation_chain_only": True,
        "gold_path_length_matches_hop": True,
        "representations": list(REPRESENTATIONS),
    }


def main() -> None:
    args = parse_args()
    counts = parse_counts(args.counts)
    for path in (args.kb, args.train):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest_path = args.output_dir / "oracle_questions.jsonl"
    if manifest_path.exists() and not args.force:
        raise RuntimeError(
            f"Frozen manifest already exists: {manifest_path}. "
            "Use --force only when intentionally replacing it."
        )

    with args.kb.open(encoding="utf-8") as handle:
        kb = json.load(handle)
    with args.train.open(encoding="utf-8") as handle:
        train = json.load(handle)

    excluded_rows: list[dict[str, Any]] = []
    if not args.include_pilot and args.exclude_manifest.exists():
        with args.exclude_manifest.open(encoding="utf-8") as handle:
            excluded_rows = [json.loads(line) for line in handle if line.strip()]
    excluded_indices = {int(row["train_index"]) for row in excluded_rows}
    excluded_by_hop = Counter(int(row["hop_count"]) for row in excluded_rows)

    rows: list[dict[str, Any]] = []
    for hop in (1, 2, 3):
        candidates = oracle.build_bundle(
            train=train,
            kb=kb,
            hops=[hop],
            per_hop=counts[hop] + excluded_by_hop[hop],
            seed=args.seed,
        )
        kept = [
            row for row in candidates if int(row["train_index"]) not in excluded_indices
        ][: counts[hop]]
        if len(kept) != counts[hop]:
            raise RuntimeError(
                f"After pilot exclusion, hop {hop} has {len(kept)} rows; "
                f"expected {counts[hop]}"
            )
        rows.extend(kept)
    validation = validate(rows, counts, excluded_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifest_path, rows)
    report = {
        "status": "complete",
        "dataset": "KQA Pro",
        "sampling": "deterministic stratified executable single-relation chains",
        "seed": args.seed,
        "pilot_exclusion_manifest": (
            str(args.exclude_manifest.resolve()) if excluded_rows else None
        ),
        "excluded_pilot_questions": len(excluded_rows),
        "excluded_pilot_questions_by_hop": {
            str(hop): excluded_by_hop[hop] for hop in (1, 2, 3)
        },
        "requested_counts": {str(hop): counts[hop] for hop in sorted(counts)},
        "source_kb": str(args.kb.resolve()),
        "source_train": str(args.train.resolve()),
        "source_kb_sha256": sha256_file(args.kb),
        "source_train_sha256": sha256_file(args.train),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "validation": validation,
    }
    write_json(args.output_dir / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
