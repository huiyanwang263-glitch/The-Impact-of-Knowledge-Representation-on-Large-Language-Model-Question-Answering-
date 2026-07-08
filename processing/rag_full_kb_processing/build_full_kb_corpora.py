#!/usr/bin/env python3
"""Build aligned Document, shuffled Triple, and Graph corpora from full KQA Pro KB."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent
WORKSPACE = DATASET_DIR.parent
PROCESSING_DIR = DATASET_DIR / "knowledge_representation_processing"
sys.path.insert(0, str(PROCESSING_DIR))

from build_pilot_representations import (  # noqa: E402
    add_statement,
    canonical_relation,
    entity_ref,
    format_value,
    is_git_lfs_pointer,
    literal_ref,
    normalize_qualifiers,
    normalize_value,
    statement_to_sentence,
)


def default_kb() -> Path:
    local = DATASET_DIR / "kb.json"
    return local if local.exists() and not is_git_lfs_pointer(local) else WORKSPACE / "kb.json"


def json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_statements(kb: dict[str, Any]) -> list[dict[str, Any]]:
    """Create one deduplicated canonical statement layer for all representations."""
    statements: dict[str, dict[str, Any]] = {}

    for concept_id, concept in kb["concepts"].items():
        for parent_id in concept.get("instanceOf", []):
            add_statement(
                statements,
                {
                    "kind": "type",
                    "subject": entity_ref(kb, concept_id),
                    "predicate": "instance of",
                    "object": entity_ref(kb, parent_id),
                    "qualifiers": [],
                },
            )

    for entity_id, entity in kb["entities"].items():
        for concept_id in entity.get("instanceOf", []):
            add_statement(
                statements,
                {
                    "kind": "type",
                    "subject": entity_ref(kb, entity_id),
                    "predicate": "instance of",
                    "object": entity_ref(kb, concept_id),
                    "qualifiers": [],
                },
            )

        for attribute in entity.get("attributes", []):
            add_statement(
                statements,
                {
                    "kind": "attribute",
                    "subject": entity_ref(kb, entity_id),
                    "predicate": attribute["key"],
                    "object": literal_ref(normalize_value(attribute["value"])),
                    "qualifiers": normalize_qualifiers(attribute.get("qualifiers")),
                },
            )

        for relation in entity.get("relations", []):
            subject_id, predicate, object_id = canonical_relation(entity_id, relation)
            if subject_id not in kb["entities"] and subject_id not in kb["concepts"]:
                continue
            if object_id not in kb["entities"] and object_id not in kb["concepts"]:
                continue
            add_statement(
                statements,
                {
                    "kind": "relation",
                    "subject": entity_ref(kb, subject_id),
                    "predicate": predicate,
                    "object": entity_ref(kb, object_id),
                    "qualifiers": normalize_qualifiers(relation.get("qualifiers")),
                },
            )

    return sorted(
        statements.values(),
        key=lambda item: (
            item["subject"]["id"],
            item["predicate"].casefold(),
            item["object"]["label"].casefold(),
            item["statement_id"],
        ),
    )


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def qualifier_suffix(qualifiers: list[dict[str, Any]]) -> str:
    if not qualifiers:
        return ""
    text = "; ".join(
        f"{item['key']}={format_value(item['value'])}" for item in qualifiers
    )
    return f" [qualifiers: {text}]"


def triple_text(statement: dict[str, Any]) -> str:
    return (
        f"({statement['subject']['label']} | {statement['predicate']} | "
        f"{statement['object']['label']}){qualifier_suffix(statement['qualifiers'])}"
    )


def build_documents(
    statements: list[dict[str, Any]], chunk_size: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for statement in statements:
        grouped[statement["subject"]["id"]].append(statement)

    documents: list[dict[str, Any]] = []
    for subject_id in sorted(grouped):
        subject_statements = grouped[subject_id]
        label = subject_statements[0]["subject"]["label"]
        for chunk_index, part in enumerate(chunks(subject_statements, chunk_size)):
            sentences = [statement_to_sentence(item) for item in part]
            documents.append(
                {
                    "chunk_id": f"document_{subject_id}_{chunk_index:04d}",
                    "subject_id": subject_id,
                    "subject_label": label,
                    "chunk_index": chunk_index,
                    "statement_ids": [item["statement_id"] for item in part],
                    "text": f"Knowledge about {label}: " + " ".join(sentences),
                }
            )
    return documents


def build_triples(
    statements: list[dict[str, Any]], shuffle_seed: int
) -> list[dict[str, Any]]:
    shuffled = list(statements)
    random.Random(shuffle_seed).shuffle(shuffled)
    return [
        {
            "chunk_id": f"triple_{item['statement_id']}",
            "statement_id": item["statement_id"],
            "subject": item["subject"],
            "predicate": item["predicate"],
            "object": item["object"],
            "qualifiers": item["qualifiers"],
            "text": triple_text(item),
        }
        for item in shuffled
    ]


def build_graph(
    statements: list[dict[str, Any]], adjacency_chunk_size: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    grouped_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for statement in statements:
        subject = statement["subject"]
        nodes[subject["id"]] = subject
        obj = statement["object"]
        if obj["kind"] == "literal":
            target_id = f"literal_{statement['statement_id']}"
            nodes[target_id] = {"id": target_id, **obj}
        else:
            target_id = obj["id"]
            nodes[target_id] = obj
        edge = {
            "edge_id": statement["statement_id"],
            "source": subject["id"],
            "source_label": subject["label"],
            "relation": statement["predicate"],
            "target": target_id,
            "target_label": obj["label"],
            "kind": statement["kind"],
            "qualifiers": statement["qualifiers"],
        }
        edges.append(edge)
        grouped_edges[subject["id"]].append(edge)

    adjacency: list[dict[str, Any]] = []
    for source_id in sorted(grouped_edges):
        source_edges = grouped_edges[source_id]
        source_label = source_edges[0]["source_label"]
        for chunk_index, part in enumerate(chunks(source_edges, adjacency_chunk_size)):
            lines = [
                f"EDGE {edge['source']} ({edge['source_label']}) "
                f"--[{edge['relation']}]--> {edge['target']} ({edge['target_label']})"
                + qualifier_suffix(edge["qualifiers"])
                for edge in part
            ]
            adjacency.append(
                {
                    "chunk_id": f"graph_{source_id}_{chunk_index:04d}",
                    "source_id": source_id,
                    "source_label": source_label,
                    "chunk_index": chunk_index,
                    "statement_ids": [edge["edge_id"] for edge in part],
                    "text": "Directed labelled graph (one edge per line):\n"
                    + "\n".join(lines),
                }
            )
    return sorted(nodes.values(), key=lambda node: node["id"]), edges, adjacency


def digest_ids(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, default=default_kb())
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "full_kb")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if is_git_lfs_pointer(args.kb):
        raise RuntimeError(f"{args.kb} is a Git LFS pointer, not the real KB")
    with args.kb.open(encoding="utf-8") as handle:
        kb = json.load(handle)

    print("Building canonical statements...", flush=True)
    statements = build_statements(kb)
    print(f"Canonical statements: {len(statements)}", flush=True)
    documents = build_documents(statements, args.chunk_size)
    triples = build_triples(statements, args.shuffle_seed)
    nodes, edges, adjacency = build_graph(statements, args.chunk_size)

    canonical_ids = {item["statement_id"] for item in statements}
    document_ids = {item for row in documents for item in row["statement_ids"]}
    triple_ids = {row["statement_id"] for row in triples}
    graph_ids = {row["edge_id"] for row in edges}
    if not canonical_ids == document_ids == triple_ids == graph_ids:
        raise RuntimeError("Statement IDs are not aligned across representations")

    output = args.output_dir
    write_jsonl(output / "canonical" / "statements.jsonl", statements)
    write_jsonl(output / "document" / "documents.jsonl", documents)
    write_jsonl(output / "triple" / "unordered_triples.jsonl", triples)
    write_jsonl(output / "graph" / "nodes.jsonl", nodes)
    write_jsonl(output / "graph" / "edges.jsonl", edges)
    write_jsonl(output / "graph" / "adjacency_chunks.jsonl", adjacency)

    manifest = {
        "purpose": "Full-KB RAG corpora; no question answers or gold subgraphs used",
        "source_kb": str(args.kb.resolve()),
        "chunk_size": args.chunk_size,
        "graph_serialization": (
            "EDGE source_id (source_label) --[predicate]--> target_id (target_label)"
        ),
        "shuffle_seed": args.shuffle_seed,
        "counts": {
            "entities": len(kb["entities"]),
            "concepts": len(kb["concepts"]),
            "canonical_statements": len(statements),
            "document_chunks": len(documents),
            "triple_chunks": len(triples),
            "graph_nodes": len(nodes),
            "graph_edges": len(edges),
            "graph_adjacency_chunks": len(adjacency),
        },
        "alignment": {
            "passed": True,
            "statement_ids_sha256": digest_ids(canonical_ids),
            "document_statement_ids_sha256": digest_ids(document_ids),
            "triple_statement_ids_sha256": digest_ids(triple_ids),
            "graph_edge_ids_sha256": digest_ids(graph_ids),
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
