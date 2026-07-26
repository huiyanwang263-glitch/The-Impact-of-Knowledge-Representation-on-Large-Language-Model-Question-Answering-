#!/usr/bin/env python3
"""Build deterministic one-question-per-hop smoke manifests from formal inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FORMAL_INPUTS = HERE / "output" / "inputs"
DEFAULT_OUTPUT = HERE / "output" / "smoke" / "deepinfra" / "inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-hop", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_per_hop(rows: list[dict[str, Any]], per_hop: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[int] = Counter()
    for row in rows:
        hop = int(row["hop_count"])
        if hop in {1, 2, 3} and counts[hop] < per_hop:
            selected.append(row)
            counts[hop] += 1
    expected = {1: per_hop, 2: per_hop, 3: per_hop}
    if dict(counts) != expected:
        raise RuntimeError(f"Expected smoke counts {expected}, found {dict(counts)}")
    return selected


def main() -> None:
    args = parse_args()
    if args.per_hop <= 0:
        raise ValueError("--per-hop must be positive")
    sources = {
        "kqapro": (
            FORMAL_INPUTS / "kqapro_100_100_60" / "oracle_questions.jsonl",
            "oracle_questions.jsonl",
        ),
        "grailqa": (
            FORMAL_INPUTS / "grailqa_100_100_60" / "selected_questions.jsonl",
            "selected_questions.jsonl",
        ),
    }
    report: dict[str, Any] = {"per_hop": args.per_hop, "datasets": {}}
    for dataset, (source, filename) in sources.items():
        rows = select_per_hop(read_jsonl(source), args.per_hop)
        destination = args.output_dir / dataset / filename
        write_jsonl(destination, rows)
        report["datasets"][dataset] = {
            "source": str(source.resolve()),
            "output": str(destination.resolve()),
            "question_ids": [
                str(row.get("question_id", row.get("qid"))) for row in rows
            ],
            "by_hop": {
                str(hop): sum(int(row["hop_count"]) == hop for row in rows)
                for hop in (1, 2, 3)
            },
        }
    report_path = args.output_dir / "build_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
