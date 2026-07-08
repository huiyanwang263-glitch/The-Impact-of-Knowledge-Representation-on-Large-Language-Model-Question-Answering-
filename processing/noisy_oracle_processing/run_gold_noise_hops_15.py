#!/usr/bin/env python3
"""Gold + Noise experiment on the fixed 1/2/3-hop KQA Pro questions."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DATASET = HERE.parent
WORKSPACE = DATASET.parent
ORACLE_DIR = DATASET / "knowledge_representation_processing"
RAG_DIR = DATASET / "rag_full_kb_processing"
sys.path.insert(0, str(ORACLE_DIR))

import run_oracle_20 as oracle  # noqa: E402
from build_pilot_representations import format_value, statement_to_sentence  # noqa: E402


REPS = ("document", "triple", "graph")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qualifier_text(qualifiers: list[dict[str, Any]]) -> str:
    if not qualifiers:
        return ""
    rendered = "; ".join(
        f"{item['key']}={format_value(item['value'])}" for item in qualifiers
    )
    return f" [qualifiers: {rendered}]"


def triple_line(statement: dict[str, Any]) -> str:
    return (
        f"({statement['subject']['label']} | {statement['predicate']} | "
        f"{statement['object']['label']}){qualifier_text(statement['qualifiers'])}"
    )


def render_contexts(statements: list[dict[str, Any]], seed: int) -> dict[str, str]:
    ordered = list(statements)
    random.Random(seed).shuffle(ordered)
    document = " ".join(statement_to_sentence(item) for item in ordered)

    triple_order = list(ordered)
    random.Random(seed + 1).shuffle(triple_order)
    triples = "\n".join(triple_line(item) for item in triple_order)

    edges = ["Directed labelled graph (one edge per line):"]
    for statement in ordered:
        source_id = statement["subject"]["id"]
        source_label = statement["subject"]["label"]
        if statement["object"]["kind"] == "literal":
            target_id = f"literal_{statement['statement_id']}"
        else:
            target_id = statement["object"]["id"]
        target_label = statement["object"]["label"]
        edges.append(
            f"EDGE {source_id} ({source_label}) --[{statement['predicate']}]--> "
            f"{target_id} ({target_label})"
            + qualifier_text(statement["qualifiers"])
        )
    return {"document": document, "triple": triples, "graph": "\n".join(edges)}


def relevant_keys(questions: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    gold_ids, entities, predicates = set(), set(), set()
    for question in questions:
        for statement in question["gold_statements"]:
            gold_ids.add(statement["statement_id"])
            predicates.add(statement["predicate"])
            entities.add(statement["subject"]["id"])
            if statement["object"]["kind"] != "literal":
                entities.add(statement["object"]["id"])
    return gold_ids, entities, predicates


def load_noise_pool(
    canonical_path: Path,
    questions: list[dict[str, Any]],
    reservoir_size: int,
    seed: int,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[str]],
    dict[str, list[str]],
    list[str],
]:
    """Scan full KB once, retaining hard candidates and a random fallback reservoir."""
    gold_ids, relevant_entities, relevant_predicates = relevant_keys(questions)
    statements: dict[str, dict[str, Any]] = {}
    by_entity: dict[str, list[str]] = defaultdict(list)
    by_predicate: dict[str, list[str]] = defaultdict(list)
    reservoir: list[str] = []
    hard_ids: set[str] = set()
    rng = random.Random(seed)
    seen_fallback = 0

    for statement in read_jsonl(canonical_path):
        statement_id = statement["statement_id"]
        if statement_id in gold_ids:
            statements[statement_id] = statement
            continue
        entity_ids = {statement["subject"]["id"]}
        if statement["object"]["kind"] != "literal":
            entity_ids.add(statement["object"]["id"])
        matched_entities = entity_ids & relevant_entities
        matched_predicate = statement["predicate"] in relevant_predicates
        if matched_entities or matched_predicate:
            statements[statement_id] = statement
            hard_ids.add(statement_id)
            for entity_id in matched_entities:
                by_entity[entity_id].append(statement_id)
            if matched_predicate:
                by_predicate[statement["predicate"]].append(statement_id)

        seen_fallback += 1
        if len(reservoir) < reservoir_size:
            reservoir.append(statement_id)
            statements[statement_id] = statement
        else:
            position = rng.randrange(seen_fallback)
            if position < reservoir_size:
                old_id = reservoir[position]
                reservoir[position] = statement_id
                statements[statement_id] = statement
                # Do not delete old hard candidates that are referenced by an index.
                if old_id not in hard_ids and old_id not in gold_ids:
                    statements.pop(old_id, None)
    return statements, dict(by_entity), dict(by_predicate), reservoir


def choose_noise(
    question: dict[str, Any],
    statements: dict[str, dict[str, Any]],
    by_entity: dict[str, list[str]],
    by_predicate: dict[str, list[str]],
    reservoir: list[str],
    noise_per_gold: int,
    seed: int,
) -> list[dict[str, Any]]:
    gold_ids = set(question["gold_statement_ids"])
    selected: list[str] = []
    selected_set = set(gold_ids)

    for gold_index, gold in enumerate(question["gold_statements"]):
        entity_ids = [gold["subject"]["id"]]
        if gold["object"]["kind"] != "literal":
            entity_ids.append(gold["object"]["id"])
        hard_entity = [sid for entity_id in entity_ids for sid in by_entity.get(entity_id, [])]
        same_predicate = list(by_predicate.get(gold["predicate"], []))
        rng = random.Random(seed + question["train_index"] * 101 + gold_index)
        rng.shuffle(hard_entity)
        rng.shuffle(same_predicate)
        fallback = list(reservoir)
        rng.shuffle(fallback)
        needed = noise_per_gold
        for candidate_id in hard_entity + same_predicate + fallback:
            if candidate_id in selected_set or candidate_id not in statements:
                continue
            selected.append(candidate_id)
            selected_set.add(candidate_id)
            needed -= 1
            if needed == 0:
                break
        if needed:
            raise RuntimeError(
                f"Not enough noise for {question['question_id']} gold statement {gold_index}"
            )
    return [statements[statement_id] for statement_id in selected]


def build_bundle(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = list(read_jsonl(args.questions))
    statements, by_entity, by_predicate, reservoir = load_noise_pool(
        args.canonical_statements,
        questions,
        args.reservoir_size,
        args.noise_seed,
    )
    bundle = []
    for question in questions:
        noise = choose_noise(
            question,
            statements,
            by_entity,
            by_predicate,
            reservoir,
            args.noise_per_gold,
            args.noise_seed,
        )
        gold = question["gold_statements"]
        combined = list(gold) + noise
        contexts = render_contexts(combined, args.noise_seed + question["train_index"])
        bundle.append(
            {
                "question_id": question["question_id"],
                "train_index": question["train_index"],
                "hop_count": question["hop_count"],
                "question": question["question"],
                "choices": question["choices"],
                "answer": question["answer"],
                "gold_statement_ids": question["gold_statement_ids"],
                "noise_statement_ids": [item["statement_id"] for item in noise],
                "gold_statements": gold,
                "noise_statements": noise,
                "noise_per_gold": args.noise_per_gold,
                "contexts": contexts,
            }
        )
    return bundle


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(row["correct"] for row in rows)
    invalid = sum(row["prediction"] is None for row in rows)
    valid = len(rows) - invalid
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows) if rows else None,
        "invalid_outputs": invalid,
        "invalid_rate": invalid / len(rows) if rows else None,
        "accuracy_on_valid_outputs": correct / valid if valid else None,
    }


def build_metrics(
    bundle: list[dict[str, Any]], predictions: list[dict[str, Any]], model: str
) -> dict[str, Any]:
    hops = sorted({item["hop_count"] for item in bundle})
    return {
        "status": "completed",
        "condition": "gold_plus_noise",
        "model": model,
        "questions": len(bundle),
        "calls": len(predictions),
        "noise_per_gold": bundle[0]["noise_per_gold"],
        "by_representation": {
            rep: summary([row for row in predictions if row["representation"] == rep])
            for rep in REPS
        },
        "by_hop": {
            str(hop): {
                rep: summary(
                    [
                        row
                        for row in predictions
                        if row["hop_count"] == hop and row["representation"] == rep
                    ]
                )
                for rep in REPS
            }
            for hop in hops
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ORACLE_DIR / "output" / "oracle_hops_15" / "oracle_questions.jsonl",
    )
    parser.add_argument(
        "--canonical-statements",
        type=Path,
        default=RAG_DIR / "output" / "full_kb" / "canonical" / "statements.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "noise_1_to_5")
    parser.add_argument("--noise-per-gold", type=int, default=5)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--reservoir-size", type=int, default=10000)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle.write_jsonl(args.output_dir / "noise_questions.jsonl", bundle)
    validation = {
        "condition": "gold_plus_noise",
        "questions": len(bundle),
        "noise_per_gold": args.noise_per_gold,
        "noise_seed": args.noise_seed,
        "graph_serialization": (
            "EDGE source_id (source_label) --[predicate]--> target_id (target_label)"
        ),
        "same_statement_ids_for_all_representations": True,
        "gold_present_in_every_context": True,
        "existing_experiments_modified": False,
    }
    oracle.write_json(args.output_dir / "bundle_validation.json", validation)
    if args.build_only or not (args.base_url and args.model):
        oracle.write_json(
            args.output_dir / "metrics.json",
            {"status": "build_only", "llm_accuracy": None, **validation},
        )
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return

    predictions = []
    for number, item in enumerate(bundle, 1):
        for rep in REPS:
            raw, usage = oracle.call_chat_completions(
                args.base_url,
                args.api_key,
                args.model,
                oracle.make_prompt(item, rep),
                args.timeout,
                args.retries,
                args.max_tokens,
            )
            prediction = oracle.extract_choice(raw, item["choices"])
            predictions.append(
                {
                    "question_id": item["question_id"],
                    "hop_count": item["hop_count"],
                    "representation": rep,
                    "gold_answer": item["answer"],
                    "prediction": prediction,
                    "raw_response": raw,
                    "correct": oracle.normalized_equal(prediction, item["answer"]),
                    "usage": usage,
                }
            )
        print(f"Completed {number}/{len(bundle)} ({item['hop_count']}-hop)", flush=True)
    oracle.write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    report = build_metrics(bundle, predictions, args.model)
    oracle.write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
