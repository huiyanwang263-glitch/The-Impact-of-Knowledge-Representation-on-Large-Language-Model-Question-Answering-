#!/usr/bin/env python3
"""Build statement-aligned atomic Document, Triple, and Graph RAG corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DATASET = HERE.parent
PROCESSING = DATASET / "knowledge_representation_processing"
sys.path.insert(0, str(PROCESSING))

from build_pilot_representations import format_value, statement_to_sentence  # noqa: E402


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def qualifier_text(qualifiers: list[dict[str, Any]]) -> str:
    if not qualifiers:
        return ""
    rendered = "; ".join(
        f"{item['key']}={format_value(item['value'])}" for item in qualifiers
    )
    return f" [qualifiers: {rendered}]"


def triple_text(statement: dict[str, Any]) -> str:
    return (
        f"({statement['subject']['label']} | {statement['predicate']} | "
        f"{statement['object']['label']}){qualifier_text(statement['qualifiers'])}"
    )


def graph_text(statement: dict[str, Any]) -> str:
    if statement["object"]["kind"] == "literal":
        target_id = f"literal_{statement['statement_id']}"
    else:
        target_id = statement["object"]["id"]
    return (
        f"EDGE {statement['subject']['id']} ({statement['subject']['label']}) "
        f"--[{statement['predicate']}]--> {target_id} ({statement['object']['label']})"
        f"{qualifier_text(statement['qualifiers'])}"
    )


def atomic_row(statement: dict[str, Any], representation: str) -> dict[str, Any]:
    statement_id = statement["statement_id"]
    if representation == "document":
        text = statement_to_sentence(statement)
    elif representation == "triple":
        text = triple_text(statement)
    else:
        text = graph_text(statement)
    return {
        "chunk_id": f"atomic_{statement_id}",
        "statement_id": statement_id,
        "statement_ids": [statement_id],
        "text": text,
    }


def digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-statements",
        type=Path,
        default=HERE / "output" / "full_kb" / "canonical" / "statements.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "atomic_kb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statements = list(read_jsonl(args.canonical_statements))
    if not statements:
        raise RuntimeError(f"No canonical statements: {args.canonical_statements}")
    statement_ids = [row["statement_id"] for row in statements]
    if len(statement_ids) != len(set(statement_ids)):
        raise RuntimeError("Canonical statement IDs are not unique")

    counts = {}
    for representation in ("document", "triple", "graph"):
        path = args.output_dir / representation / "atomic_statements.jsonl"
        counts[representation] = write_jsonl(
            path, (atomic_row(statement, representation) for statement in statements)
        )
    if len(set(counts.values())) != 1:
        raise AssertionError("Atomic corpora are not aligned")

    manifest = {
        "purpose": "Controlled E3 RAG with one canonical statement per aligned chunk",
        "source": str(args.canonical_statements.resolve()),
        "chunk_unit": "one_statement",
        "shared_chunk_ids_across_representations": True,
        "graph_serialization": (
            "EDGE source_id (source_label) --[predicate]--> target_id (target_label)"
        ),
        "counts": counts,
        "statement_ids_sha256": digest(statement_ids),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
