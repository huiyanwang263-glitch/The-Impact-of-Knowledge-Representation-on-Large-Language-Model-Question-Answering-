#!/usr/bin/env python3
"""Build a frozen, auditable alias cache for the GrailQA evaluation answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
GRAILQA_DATA = PROJECT / "GrailQA_v1.0 (1)" / "GrailQA_v1.0"
GRAILQA_CODE = PROJECT / "GrailQA_v1.0 (1)" / "GrailQA"
DEFAULT_QUESTIONS = (
    HERE.parent
    / "grailqa_mini_kb_processing"
    / "output"
    / "dev_10_per_hop"
    / "selected_questions.jsonl"
)
DEFAULT_DATASETS = [
    GRAILQA_DATA / "grailqa_v1.0_train.json",
    GRAILQA_DATA / "grailqa_v1.0_dev.json",
]
DEFAULT_ENTITY_LINKING = GRAILQA_CODE / "entity_linking_results" / "grailqa_el.json"
DEFAULT_OUTPUT = HERE / "output" / "answer_alias_cache.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = value.replace("–", "-").replace("—", "-")
    value = "".join(" " if char in string.punctuation else char for char in value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def target_answers(
    rows: list[dict[str, Any]],
) -> tuple[set[str], dict[str, set[str]], Counter[str]]:
    target_ids: set[str] = set()
    canonical: dict[str, set[str]] = defaultdict(set)
    by_hop: Counter[str] = Counter()
    for row in rows:
        by_hop[str(row["hop_count"])] += 1
        answers = row.get("answers", [])
        for answer in answers:
            mid = str(answer["answer_argument"])
            target_ids.add(mid)
            name = str(answer.get("entity_name") or "").strip()
            if name:
                canonical[mid].add(name)
    return target_ids, canonical, by_hop


def add_alias(
    store: dict[str, dict[str, dict[str, Any]]],
    mid: str,
    text: str,
    source: str,
    count: int = 1,
) -> None:
    text = str(text).strip()
    normalized = normalize_answer(text)
    if not text or not normalized:
        return
    entry = store[mid].setdefault(
        normalized,
        {"normalized": normalized, "surface_forms": set(), "source_counts": Counter()},
    )
    entry["surface_forms"].add(text)
    entry["source_counts"][source] += count


def collect_dataset_aliases(
    path: Path,
    target_ids: set[str],
    store: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, int]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected a JSON array in {path}")
    answer_hits = 0
    graph_node_hits = 0
    for row in rows:
        for answer in row.get("answer", []):
            mid = str(answer.get("answer_argument", ""))
            if mid in target_ids and answer.get("entity_name"):
                add_alias(store, mid, answer["entity_name"], "grailqa_answer_entity_name")
                answer_hits += 1
        for node in row.get("graph_query", {}).get("nodes", []):
            mid = str(node.get("id", ""))
            if mid in target_ids and node.get("friendly_name"):
                add_alias(
                    store,
                    mid,
                    node["friendly_name"],
                    "grailqa_graph_node_friendly_name",
                )
                graph_node_hits += 1
    return {
        "rows": len(rows),
        "answer_name_hits": answer_hits,
        "graph_node_name_hits": graph_node_hits,
    }


def iter_linked_entities(value: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        entities = value.get("entities")
        if isinstance(entities, dict):
            for mid, metadata in entities.items():
                if isinstance(metadata, dict):
                    yield str(mid), metadata
        for nested in value.values():
            yield from iter_linked_entities(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_linked_entities(nested)


def collect_entity_linking_aliases(
    path: Path,
    target_ids: set[str],
    store: dict[str, dict[str, dict[str, Any]]],
    min_mention_count: int,
) -> dict[str, int]:
    data = read_json(path)
    mentions: Counter[tuple[str, str]] = Counter()
    mention_surfaces: dict[tuple[str, str], set[str]] = defaultdict(set)
    friendly_names: Counter[tuple[str, str]] = Counter()
    friendly_surfaces: dict[tuple[str, str], set[str]] = defaultdict(set)
    linked_records = 0
    for mid, metadata in iter_linked_entities(data):
        if mid not in target_ids:
            continue
        linked_records += 1
        mention = str(metadata.get("mention") or "").strip()
        normalized_mention = normalize_answer(mention)
        if normalized_mention:
            key = (mid, normalized_mention)
            mentions[key] += 1
            mention_surfaces[key].add(mention)
        friendly = str(metadata.get("friendly_name") or "").strip()
        normalized_friendly = normalize_answer(friendly)
        if normalized_friendly:
            key = (mid, normalized_friendly)
            friendly_names[key] += 1
            friendly_surfaces[key].add(friendly)

    accepted_mentions = 0
    rejected_mentions = 0
    for (mid, _), count in mentions.items():
        surfaces = mention_surfaces[(mid, _)]
        if count < min_mention_count:
            rejected_mentions += 1
            continue
        for surface in surfaces:
            add_alias(store, mid, surface, "entity_linking_mention", count)
        accepted_mentions += 1
    for (mid, normalized), count in friendly_names.items():
        for surface in friendly_surfaces[(mid, normalized)]:
            add_alias(store, mid, surface, "entity_linking_friendly_name", count)
    return {
        "target_linked_records": linked_records,
        "unique_mentions_accepted": accepted_mentions,
        "unique_mentions_rejected_below_threshold": rejected_mentions,
        "unique_friendly_names": len(friendly_names),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--datasets", type=Path, nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--entity-linking", type=Path, default=DEFAULT_ENTITY_LINKING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-linker-mention-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.questions, *args.datasets, args.entity_linking]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.min_linker_mention_count < 1:
        raise ValueError("--min-linker-mention-count must be positive")

    questions = read_jsonl(args.questions)
    target_ids, canonical, by_hop = target_answers(questions)
    if not target_ids:
        raise RuntimeError("No target answer MIDs found")

    aliases: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for mid in sorted(target_ids):
        add_alias(aliases, mid, mid, "freebase_mid")
        for name in sorted(canonical.get(mid, set())):
            add_alias(aliases, mid, name, "selected_gold_entity_name")

    dataset_audit = {
        str(path.resolve()): collect_dataset_aliases(path, target_ids, aliases)
        for path in args.datasets
    }
    linker_audit = collect_entity_linking_aliases(
        args.entity_linking,
        target_ids,
        aliases,
        args.min_linker_mention_count,
    )

    normalized_to_mids: dict[str, set[str]] = defaultdict(set)
    for mid, entries in aliases.items():
        for normalized in entries:
            normalized_to_mids[normalized].add(mid)
    collisions = {
        alias: sorted(mids)
        for alias, mids in sorted(normalized_to_mids.items())
        if len(mids) > 1
    }

    entities: dict[str, Any] = {}
    for mid in sorted(target_ids):
        entries = []
        for normalized, raw in sorted(aliases[mid].items()):
            entries.append(
                {
                    "normalized": normalized,
                    "surface_forms": sorted(raw["surface_forms"], key=str.casefold),
                    "source_counts": dict(sorted(raw["source_counts"].items())),
                    "globally_ambiguous_with_target_mids": collisions.get(normalized, []),
                }
            )
        entities[mid] = {
            "canonical_names": sorted(canonical.get(mid, set()), key=str.casefold),
            "accepted_normalized_aliases": [row["normalized"] for row in entries],
            "aliases": entries,
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "normalization": "NFKC + casefold + punctuation/articles removal + whitespace",
        "target_questions": len(questions),
        "target_questions_by_hop": dict(sorted(by_hop.items())),
        "target_answer_mid_count": len(target_ids),
        "source_questions": str(args.questions.resolve()),
        "sources": {
            "datasets": [str(path.resolve()) for path in args.datasets],
            "entity_linking": str(args.entity_linking.resolve()),
        },
        "controls": {
            "min_linker_mention_count": args.min_linker_mention_count,
            "aliases_are_evaluation_only": True,
            "aliases_must_not_be_loaded_into_model_prompt": True,
        },
        "audit": {
            "datasets": dataset_audit,
            "entity_linking": linker_audit,
            "normalized_alias_count": sum(len(row["accepted_normalized_aliases"]) for row in entities.values()),
            "globally_ambiguous_normalized_alias_count": len(collisions),
            "globally_ambiguous_normalized_aliases": collisions,
        },
        "entities": entities,
    }
    payload["cache_sha256"] = sha256_value(payload)
    write_target = args.output
    write_target.parent.mkdir(parents=True, exist_ok=True)
    write_target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(write_target.resolve()),
                "cache_sha256": payload["cache_sha256"],
                "target_questions": len(questions),
                "target_answer_mid_count": len(target_ids),
                "normalized_alias_count": payload["audit"]["normalized_alias_count"],
                "globally_ambiguous_normalized_alias_count": len(collisions),
                "entity_linking": linker_audit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

