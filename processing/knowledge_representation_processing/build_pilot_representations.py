#!/usr/bin/env python3
"""Build three semantically aligned representations from a KQA Pro KB subgraph.

The script first creates a canonical statement layer, then renders exactly the
same statement IDs as natural-language documents, shuffled triples, and a
readable node-edge graph. It uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = DATASET_DIR.parent


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    return path.read_text(encoding="utf-8", errors="ignore").startswith(
        "version https://git-lfs.github.com/spec/v1"
    )


def default_kb_path() -> Path:
    local = DATASET_DIR / "kb.json"
    workspace = WORKSPACE_DIR / "kb.json"
    if local.exists() and not is_git_lfs_pointer(local):
        return local
    return workspace


def json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_statement_id(payload: dict[str, Any]) -> str:
    digest = hashlib.sha1(json_key(payload).encode("utf-8")).hexdigest()[:14]
    return f"stmt_{digest}"


def normalize_value(raw: dict[str, Any]) -> dict[str, Any]:
    value = {
        "type": raw.get("type", "string"),
        "value": raw.get("value"),
    }
    if raw.get("unit") not in (None, ""):
        value["unit"] = raw["unit"]
    return value


def normalize_qualifiers(raw: dict[str, list[dict[str, Any]]] | None) -> list[dict[str, Any]]:
    qualifiers: list[dict[str, Any]] = []
    for key, values in sorted((raw or {}).items()):
        for value in sorted((normalize_value(v) for v in values), key=json_key):
            qualifiers.append({"key": key, "value": value})
    return qualifiers


def format_value(value: dict[str, Any]) -> str:
    value_type = value.get("type")
    raw = value.get("value")
    if value_type == "date" and isinstance(raw, str):
        rendered = raw.replace("/", "-")
    else:
        rendered = str(raw)
    unit = value.get("unit")
    if unit and unit != "1":
        rendered = f"{rendered} {unit}"
    return rendered


def label_for(kb: dict[str, Any], node_id: str) -> str:
    if node_id in kb["entities"]:
        return kb["entities"][node_id].get("name", node_id)
    if node_id in kb["concepts"]:
        return kb["concepts"][node_id].get("name", node_id)
    return node_id


def node_kind(kb: dict[str, Any], node_id: str) -> str:
    if node_id in kb["entities"]:
        return "entity"
    if node_id in kb["concepts"]:
        return "concept"
    return "unknown"


def canonical_relation(owner_id: str, relation: dict[str, Any]) -> tuple[str, str, str]:
    object_id = relation["object"]
    if relation.get("direction", "forward") == "backward":
        return object_id, relation["predicate"], owner_id
    return owner_id, relation["predicate"], object_id


def relation_sort_key(kb: dict[str, Any], owner_id: str, relation: dict[str, Any]) -> tuple[str, str, str]:
    subject_id, predicate, object_id = canonical_relation(owner_id, relation)
    neighbor_id = object_id if subject_id == owner_id else subject_id
    return predicate.casefold(), label_for(kb, neighbor_id).casefold(), neighbor_id


def select_connected_entities(
    kb: dict[str, Any],
    seed_id: str,
    max_entities: int,
    expansion_relations_per_entity: int,
    max_expansion_degree: int,
) -> list[str]:
    if seed_id not in kb["entities"]:
        raise ValueError(f"Seed entity {seed_id!r} does not exist in kb.json")

    selected: list[str] = [seed_id]
    selected_set = {seed_id}
    queue: deque[str] = deque([seed_id])

    while queue and len(selected) < max_entities:
        owner_id = queue.popleft()
        # Keep hubs (countries, large regions, etc.) as useful leaf nodes, but
        # do not let them dominate a small topical pilot through thousands of
        # weakly related neighbours.
        if len(kb["entities"][owner_id].get("relations", [])) > max_expansion_degree:
            continue
        relations = sorted(
            kb["entities"][owner_id].get("relations", []),
            key=lambda relation: relation_sort_key(kb, owner_id, relation),
        )
        expanded = 0
        for relation in relations:
            subject_id, _, object_id = canonical_relation(owner_id, relation)
            neighbor_id = object_id if subject_id == owner_id else subject_id
            if neighbor_id not in kb["entities"]:
                continue
            expanded += 1
            if neighbor_id not in selected_set:
                selected.append(neighbor_id)
                selected_set.add(neighbor_id)
                queue.append(neighbor_id)
                if len(selected) >= max_entities:
                    break
            if expanded >= expansion_relations_per_entity:
                break

    return selected


def entity_ref(kb: dict[str, Any], node_id: str) -> dict[str, str]:
    return {
        "id": node_id,
        "label": label_for(kb, node_id),
        "kind": node_kind(kb, node_id),
    }


def literal_ref(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "literal",
        "label": format_value(value),
        **value,
    }


def add_statement(target: dict[str, dict[str, Any]], payload: dict[str, Any]) -> None:
    identity = {
        "kind": payload["kind"],
        "subject": payload["subject"],
        "predicate": payload["predicate"],
        "object": payload["object"],
        "qualifiers": payload["qualifiers"],
    }
    statement_id = stable_statement_id(identity)
    if statement_id not in target:
        target[statement_id] = {"statement_id": statement_id, **payload}


def build_canonical_statements(
    kb: dict[str, Any],
    entity_ids: list[str],
    max_attributes_per_entity: int,
) -> list[dict[str, Any]]:
    selected = set(entity_ids)
    statements: dict[str, dict[str, Any]] = {}

    for entity_id in entity_ids:
        entity = kb["entities"][entity_id]

        for concept_id in sorted(entity.get("instanceOf", []), key=lambda x: label_for(kb, x).casefold()):
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

        attributes = sorted(
            entity.get("attributes", []),
            key=lambda attr: (attr["key"].casefold(), json_key(normalize_value(attr["value"]))),
        )[:max_attributes_per_entity]
        for attribute in attributes:
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
            if subject_id not in selected or object_id not in selected:
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
            item["subject"]["label"].casefold(),
            item["predicate"].casefold(),
            item["object"]["label"].casefold(),
            item["statement_id"],
        ),
    )


def qualifier_text(qualifiers: list[dict[str, Any]]) -> str:
    if not qualifiers:
        return ""
    values = [f"{item['key']}: {format_value(item['value'])}" for item in qualifiers]
    return f" Qualifiers: {'; '.join(values)}."


def statement_to_sentence(statement: dict[str, Any]) -> str:
    subject = statement["subject"]["label"]
    predicate = statement["predicate"]
    obj = statement["object"]["label"]
    if statement["kind"] == "type":
        sentence = f"{subject} is an instance of {obj}."
    elif statement["kind"] == "attribute":
        sentence = f"The {predicate} of {subject} is {obj}."
    else:
        sentence = f"{subject} has the relation '{predicate}' to {obj}."
    return sentence + qualifier_text(statement["qualifiers"])


def build_documents(statements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for statement in statements:
        grouped[statement["subject"]["id"]].append(statement)

    documents: list[dict[str, Any]] = []
    for subject_id, items in sorted(
        grouped.items(), key=lambda pair: pair[1][0]["subject"]["label"].casefold()
    ):
        sentences = [statement_to_sentence(item) for item in items]
        documents.append(
            {
                "document_id": f"doc_{subject_id}",
                "subject_id": subject_id,
                "title": items[0]["subject"]["label"],
                "statement_ids": [item["statement_id"] for item in items],
                "sentences": sentences,
                "text": " ".join(sentences),
            }
        )
    return documents


def build_unordered_triples(
    statements: list[dict[str, Any]], shuffle_seed: int
) -> list[dict[str, Any]]:
    triples = [
        {
            "statement_id": item["statement_id"],
            "subject": item["subject"],
            "predicate": item["predicate"],
            "object": item["object"],
            "qualifiers": item["qualifiers"],
        }
        for item in statements
    ]
    random.Random(shuffle_seed).shuffle(triples)
    return triples


def build_graph(statements: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for statement in statements:
        subject = statement["subject"]
        nodes[subject["id"]] = subject

        obj = statement["object"]
        if obj["kind"] == "literal":
            object_id = f"literal_{statement['statement_id']}"
            nodes[object_id] = {"id": object_id, **obj}
        else:
            object_id = obj["id"]
            nodes[object_id] = obj

        edges.append(
            {
                "edge_id": statement["statement_id"],
                "source": subject["id"],
                "relation": statement["predicate"],
                "target": object_id,
                "kind": statement["kind"],
                "qualifiers": statement["qualifiers"],
            }
        )

    return {
        "nodes": sorted(nodes.values(), key=lambda node: (node["kind"], node["label"].casefold())),
        "edges": edges,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def render_documents_markdown(documents: list[dict[str, Any]]) -> str:
    blocks = ["# Natural-language documents", ""]
    for document in documents:
        blocks.extend(
            [
                f"## {document['title']}",
                "",
                document["text"],
                "",
                f"Statement IDs: {', '.join(document['statement_ids'])}",
                "",
            ]
        )
    return "\n".join(blocks)


def render_triples_text(triples: list[dict[str, Any]]) -> str:
    lines = ["# Deterministically shuffled triples", ""]
    for triple in triples:
        line = (
            f"[{triple['statement_id']}] "
            f"({triple['subject']['label']} | {triple['predicate']} | {triple['object']['label']})"
        )
        if triple["qualifiers"]:
            rendered = "; ".join(
                f"{item['key']}={format_value(item['value'])}" for item in triple["qualifiers"]
            )
            line += f" [qualifiers: {rendered}]"
        lines.append(line)
    return "\n".join(lines) + "\n"


def render_graph_markdown(graph: dict[str, Any]) -> str:
    labels = {node["id"]: node["label"] for node in graph["nodes"]}
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph["edges"]:
        outgoing[edge["source"]].append(edge)

    lines = ["# Readable graph", "", "Each bullet is one directed, labelled edge.", ""]
    for source_id in sorted(outgoing, key=lambda item: labels[item].casefold()):
        lines.extend([f"## {labels[source_id]} `{source_id}`", ""])
        for edge in sorted(
            outgoing[source_id], key=lambda item: (item["relation"].casefold(), labels[item["target"]].casefold())
        ):
            line = (
                f"- **{edge['relation']}** --> {labels[edge['target']]} "
                f"`{edge['edge_id']}`"
            )
            if edge["qualifiers"]:
                rendered = "; ".join(
                    f"{item['key']}={format_value(item['value'])}" for item in edge["qualifiers"]
                )
                line += f"; qualifiers: {rendered}"
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


def validate_alignment(
    statements: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    graph: dict[str, Any],
) -> None:
    canonical_ids = {item["statement_id"] for item in statements}
    document_ids = {item for document in documents for item in document["statement_ids"]}
    triple_ids = {item["statement_id"] for item in triples}
    graph_ids = {item["edge_id"] for item in graph["edges"]}
    if not canonical_ids == document_ids == triple_ids == graph_ids:
        raise RuntimeError("Representations do not contain identical statement ID sets")
    if len(triples) != len(triple_ids):
        raise RuntimeError("Duplicate statement IDs found in triples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, default=default_kb_path())
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output" / "pilot")
    parser.add_argument("--seed-id", default="Q513809", help="Default: Mighty Morphin Power Rangers")
    parser.add_argument("--max-entities", type=int, default=15)
    parser.add_argument("--expansion-relations-per-entity", type=int, default=5)
    parser.add_argument(
        "--max-expansion-degree",
        type=int,
        default=100,
        help="Keep higher-degree hubs as leaves instead of expanding through them",
    )
    parser.add_argument("--max-attributes-per-entity", type=int, default=4)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if is_git_lfs_pointer(args.kb):
        raise RuntimeError(f"{args.kb} is a Git LFS pointer, not the real kb.json")
    with args.kb.open(encoding="utf-8") as handle:
        kb = json.load(handle)

    entity_ids = select_connected_entities(
        kb,
        seed_id=args.seed_id,
        max_entities=args.max_entities,
        expansion_relations_per_entity=args.expansion_relations_per_entity,
        max_expansion_degree=args.max_expansion_degree,
    )
    statements = build_canonical_statements(
        kb,
        entity_ids=entity_ids,
        max_attributes_per_entity=args.max_attributes_per_entity,
    )
    documents = build_documents(statements)
    triples = build_unordered_triples(statements, args.shuffle_seed)
    graph = build_graph(statements)
    validate_alignment(statements, documents, triples, graph)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "canonical_statements.jsonl", statements)
    write_jsonl(args.output_dir / "documents.jsonl", documents)
    write_jsonl(args.output_dir / "unordered_triples.jsonl", triples)
    write_json(args.output_dir / "readable_graph.json", graph)
    (args.output_dir / "documents.md").write_text(
        render_documents_markdown(documents), encoding="utf-8"
    )
    (args.output_dir / "unordered_triples.txt").write_text(
        render_triples_text(triples), encoding="utf-8"
    )
    (args.output_dir / "readable_graph.md").write_text(
        render_graph_markdown(graph), encoding="utf-8"
    )

    manifest = {
        "source_kb": str(args.kb.resolve()),
        "seed": {"id": args.seed_id, "label": label_for(kb, args.seed_id)},
        "selection": {
            "max_entities": args.max_entities,
            "expansion_relations_per_entity": args.expansion_relations_per_entity,
            "max_expansion_degree": args.max_expansion_degree,
            "max_attributes_per_entity": args.max_attributes_per_entity,
            "shuffle_seed": args.shuffle_seed,
        },
        "selected_entities": [entity_ref(kb, item) for item in entity_ids],
        "counts": {
            "selected_entities": len(entity_ids),
            "canonical_statements": len(statements),
            "documents": len(documents),
            "triples": len(triples),
            "graph_nodes": len(graph["nodes"]),
            "graph_edges": len(graph["edges"]),
        },
        "alignment_check": "passed",
        "statement_ids_sha256": hashlib.sha256(
            "\n".join(sorted(item["statement_id"] for item in statements)).encode("utf-8")
        ).hexdigest(),
    }
    write_json(args.output_dir / "manifest.json", manifest)

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote pilot representations to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
