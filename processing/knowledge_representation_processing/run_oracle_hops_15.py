#!/usr/bin/env python3
"""Run a stratified Oracle test with five linear 1/2/3-hop questions each."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
sys.path.insert(0, str(HERE))
import run_oracle_20 as base  # noqa: E402


REPRESENTATIONS = ("document", "triple", "graph")
ALLOWED_FUNCTIONS = {"Find", "Relate", "FilterConcept", "And", "What"}


def relation_depth(program: list[dict[str, Any]]) -> int:
    """Count Relate operations on the longest dependency path."""
    depths: list[int] = []
    for step in program:
        parent_depth = max((depths[index] for index in step.get("dependencies", [])), default=0)
        depths.append(parent_depth + int(step.get("function") == "Relate"))
    return depths[-1] if depths else 0


def is_linear_program(sample: dict[str, Any], hops: int) -> bool:
    program = sample.get("program", [])
    functions = [step.get("function") for step in program]
    if not program or functions[0] != "Find" or functions[-1] != "What":
        return False
    if any(function not in ALLOWED_FUNCTIONS for function in functions):
        return False
    if functions.count("What") != 1:
        return False
    # Requiring total Relate count == longest relation depth excludes parallel
    # multi-branch programs while allowing an entity-identity And constraint.
    if functions.count("Relate") != hops or relation_depth(program) != hops:
        return False
    return all(
        (step["function"] == "Find" and step.get("dependencies", []) == [])
        or (step["function"] == "And" and len(step.get("dependencies", [])) == 2)
        or (step["function"] not in {"Find", "And"} and len(step.get("dependencies", [])) == 1)
        for step in program
    )


def trace_path(
    sample: dict[str, Any],
    hops: int,
    kb: dict[str, Any],
    name_index: dict[str, list[str]],
) -> list[dict[str, Any]] | None:
    """Execute a small KoPL subset while retaining relation provenance."""
    if not is_linear_program(sample, hops):
        return None
    # Each operation result maps entity ID to alternative provenance paths.
    results: list[dict[str, list[list[dict[str, Any]]]]] = []
    for step in sample["program"]:
        function = step["function"]
        dependencies = step.get("dependencies", [])
        if function == "Find":
            current = {
                entity_id: [[]] for entity_id in name_index.get(step["inputs"][0], [])
            }
        elif function == "Relate":
            predicate, direction = step["inputs"]
            current: dict[str, list[list[dict[str, Any]]]] = {}
            for entity_id, paths in results[dependencies[0]].items():
                for relation in kb["entities"][entity_id].get("relations", []):
                    if relation["predicate"] != predicate or relation.get("direction") != direction:
                        continue
                    target_id = relation["object"]
                    if target_id not in kb["entities"]:
                        continue
                    statement = base.make_relation_statement(kb, entity_id, relation)
                    target_paths = current.setdefault(target_id, [])
                    for path in paths:
                        if len(target_paths) < 50:
                            target_paths.append(path + [statement])
        elif function == "FilterConcept":
            # Gold-answer/path validation below resolves concept ambiguity. The
            # relation facts themselves remain sufficient Oracle evidence.
            current = results[dependencies[0]]
        elif function == "And":
            left = results[dependencies[0]]
            right = results[dependencies[1]]
            current = {}
            for entity_id in left.keys() & right.keys():
                combined = []
                for left_path in left[entity_id]:
                    for right_path in right[entity_id]:
                        if len(combined) < 50:
                            combined.append(left_path + right_path)
                current[entity_id] = combined
        elif function == "What":
            current = results[dependencies[0]]
        else:
            return None
        results.append(current)
        if not current:
            return None

    gold = str(sample["answer"]).strip().casefold()
    final_result = results[-1]
    valid = [
        evidence
        for target_id, alternatives in final_result.items()
        if kb["entities"][target_id]["name"].strip().casefold() == gold
        for evidence in alternatives
        if len(evidence) == hops
        and len({item["statement_id"] for item in evidence}) == hops
    ]
    if not valid:
        return None
    valid.sort(key=lambda path: [item["statement_id"] for item in path])
    return valid[0]


def build_bundle(
    train: list[dict[str, Any]],
    kb: dict[str, Any],
    hops: list[int],
    per_hop: int,
    seed: int,
) -> list[dict[str, Any]]:
    name_index = base.build_name_index(kb)
    result: list[dict[str, Any]] = []
    for hop_count in hops:
        candidates = [
            (index, sample)
            for index, sample in enumerate(train)
            if is_linear_program(sample, hop_count)
            and sample.get("answer") in sample.get("choices", [])
        ]
        random.Random(seed + hop_count).shuffle(candidates)
        rows: list[dict[str, Any]] = []
        for train_index, sample in candidates:
            evidence = trace_path(sample, hop_count, kb, name_index)
            if evidence is None:
                continue
            rows.append(
                {
                    "question_id": f"train_{train_index}",
                    "train_index": train_index,
                    "hop_count": hop_count,
                    "path_type": "single_relation_chain",
                    "question": sample["question"],
                    "choices": sample["choices"],
                    "answer": sample["answer"],
                    "program": sample["program"],
                    "gold_statements": evidence,
                    "gold_statement_ids": [item["statement_id"] for item in evidence],
                    "contexts": base.render_all(evidence, seed + train_index),
                }
            )
            if len(rows) == per_hop:
                break
        if len(rows) != per_hop:
            raise RuntimeError(
                f"Found {len(rows)} executable linear {hop_count}-hop questions; "
                f"requested {per_hop}"
            )
        result.extend(rows)
    return result


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


def metrics_for(
    bundle: list[dict[str, Any]], predictions: list[dict[str, Any]], model: str, hops: list[int]
) -> dict[str, Any]:
    by_hop: dict[str, Any] = {}
    for hop in hops:
        hop_rows = [row for row in predictions if row["hop_count"] == hop]
        question_ids = {item["question_id"] for item in bundle if item["hop_count"] == hop}
        all_three = sum(
            all(row["correct"] for row in hop_rows if row["question_id"] == question_id)
            for question_id in question_ids
        )
        by_hop[str(hop)] = {
            "questions": len(question_ids),
            "by_representation": {
                representation: summary(
                    [row for row in hop_rows if row["representation"] == representation]
                )
                for representation in REPRESENTATIONS
            },
            "all_three_correct": {
                "count": all_three,
                "rate": all_three / len(question_ids),
            },
        }
    return {
        "status": "completed",
        "model": model,
        "task_mode": "open_ended_exact_match_to_gold_choice",
        "questions": len(bundle),
        "calls": len(predictions),
        "hops": hops,
        "by_representation": {
            representation: summary(
                [row for row in predictions if row["representation"] == representation]
            )
            for representation in REPRESENTATIONS
        },
        "by_hop": by_hop,
    }


def evaluate(bundle: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list, dict]:
    predictions: list[dict[str, Any]] = []
    for number, item in enumerate(bundle, start=1):
        for representation in REPRESENTATIONS:
            raw, usage = base.call_chat_completions(
                args.base_url,
                args.api_key,
                args.model,
                base.make_prompt(item, representation),
                args.timeout,
                args.retries,
                args.max_tokens,
            )
            prediction = base.extract_choice(raw, item["choices"])
            predictions.append(
                {
                    "question_id": item["question_id"],
                    "hop_count": item["hop_count"],
                    "representation": representation,
                    "gold_answer": item["answer"],
                    "prediction": prediction,
                    "raw_response": raw,
                    "correct": base.normalized_equal(prediction, item["answer"]),
                    "usage": usage,
                }
            )
        print(f"Completed {number}/{len(bundle)} ({item['hop_count']}-hop)", flush=True)
    return predictions, metrics_for(bundle, predictions, args.model, args.hops)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, default=WORKSPACE / "kb.json")
    parser.add_argument("--questions", type=Path, default=WORKSPACE / "train.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "oracle_hops_15")
    parser.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--per-hop", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.hops = sorted(set(args.hops))
    with args.kb.open(encoding="utf-8") as handle:
        kb = json.load(handle)
    with args.questions.open(encoding="utf-8") as handle:
        train = json.load(handle)
    bundle = build_bundle(train, kb, args.hops, args.per_hop, args.sample_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.write_jsonl(args.output_dir / "oracle_questions.jsonl", bundle)
    validation = {
        "questions": len(bundle),
        "hops": args.hops,
        "questions_per_hop": args.per_hop,
        "path_type": "single_relation_chain",
        "graph_serialization": (
            "EDGE source_id (source_label) --[predicate]--> target_id (target_label)"
        ),
        "hop_counts_match_statement_counts": all(
            item["hop_count"] == len(item["gold_statement_ids"]) for item in bundle
        ),
        "previous_output_untouched": str((HERE / "output" / "oracle_20").resolve()),
    }
    base.write_json(args.output_dir / "bundle_validation.json", validation)
    if args.build_only or not (args.base_url and args.model):
        result = {"status": "build_only", "llm_accuracy": None, **validation}
        base.write_json(args.output_dir / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    predictions, result = evaluate(bundle, args)
    base.write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    base.write_json(args.output_dir / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
