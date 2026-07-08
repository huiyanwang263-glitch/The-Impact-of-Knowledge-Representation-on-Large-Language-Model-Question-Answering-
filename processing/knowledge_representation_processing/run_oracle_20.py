#!/usr/bin/env python3
"""Run a small, paired Oracle-context experiment on 20 two-hop KQA Pro questions.

The script traces simple gold KoPL programs of the form Find -> Relate ->
FilterConcept -> Relate -> FilterConcept -> What, renders the same two relation
statements as a document, shuffled triples, and a readable graph, and
optionally calls an OpenAI-compatible chat-completions endpoint. Without a
configured model it still builds and validates the 20-question experiment
bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
sys.path.insert(0, str(HERE))

from build_pilot_representations import (  # noqa: E402
    canonical_relation,
    entity_ref,
    format_value,
    normalize_qualifiers,
    stable_statement_id,
    statement_to_sentence,
)


REPRESENTATIONS = ("document", "triple", "graph")
TARGET_PROGRAM = [
    "Find",
    "Relate",
    "FilterConcept",
    "Relate",
    "FilterConcept",
    "What",
]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_relation_statement(
    kb: dict[str, Any], owner_id: str, relation: dict[str, Any]
) -> dict[str, Any]:
    subject_id, predicate, object_id = canonical_relation(owner_id, relation)
    payload = {
        "kind": "relation",
        "subject": entity_ref(kb, subject_id),
        "predicate": predicate,
        "object": entity_ref(kb, object_id),
        "qualifiers": normalize_qualifiers(relation.get("qualifiers")),
    }
    identity = {
        "kind": payload["kind"],
        "subject": payload["subject"],
        "predicate": payload["predicate"],
        "object": payload["object"],
        "qualifiers": payload["qualifiers"],
    }
    return {"statement_id": stable_statement_id(identity), **payload}


def build_name_index(kb: dict[str, Any]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for entity_id, entity in kb["entities"].items():
        index.setdefault(entity["name"], []).append(entity_id)
    return index


def trace_two_hop(
    sample: dict[str, Any],
    kb: dict[str, Any],
    name_index: dict[str, list[str]],
) -> list[dict[str, Any]] | None:
    program = sample["program"]
    if [step["function"] for step in program] != TARGET_PROGRAM:
        return None

    start_name = program[0]["inputs"][0]
    paths: list[tuple[str, list[dict[str, Any]]]] = [
        (entity_id, []) for entity_id in name_index.get(start_name, [])
    ]
    if not paths:
        return None

    for step in (item for item in program if item["function"] == "Relate"):
        predicate, direction = step["inputs"]
        expanded: list[tuple[str, list[dict[str, Any]]]] = []
        for current_id, evidence in paths:
            for relation in kb["entities"][current_id].get("relations", []):
                if relation["predicate"] != predicate or relation.get("direction") != direction:
                    continue
                target_id = relation["object"]
                if target_id not in kb["entities"]:
                    continue
                statement = make_relation_statement(kb, current_id, relation)
                expanded.append((target_id, evidence + [statement]))
        paths = expanded
        if not paths:
            return None

    gold = str(sample["answer"]).strip().casefold()
    gold_paths = [
        evidence
        for target_id, evidence in paths
        if kb["entities"][target_id]["name"].strip().casefold() == gold
    ]
    if not gold_paths:
        return None

    # Prefer a path with two distinct statements, then use a stable ID order.
    gold_paths.sort(
        key=lambda path: (
            len({item["statement_id"] for item in path}) != 2,
            [item["statement_id"] for item in path],
        )
    )
    result = gold_paths[0]
    if len(result) != 2 or len({item["statement_id"] for item in result}) != 2:
        return None
    return result


def render_document(statements: list[dict[str, Any]]) -> str:
    return " ".join(statement_to_sentence(item) for item in statements)


def render_triples(statements: list[dict[str, Any]], seed: int) -> str:
    items = list(statements)
    random.Random(seed).shuffle(items)
    lines = []
    for item in items:
        line = f"({item['subject']['label']} | {item['predicate']} | {item['object']['label']})"
        if item["qualifiers"]:
            qualifiers = "; ".join(
                f"{q['key']}={q['value']['value']}" for q in item["qualifiers"]
            )
            line += f" [qualifiers: {qualifiers}]"
        lines.append(line)
    return "\n".join(lines)


def render_graph(statements: list[dict[str, Any]]) -> str:
    lines = ["Directed labelled graph (one edge per line):"]
    for item in statements:
        if item["object"]["kind"] == "literal":
            target_id = f"literal_{item['statement_id']}"
        else:
            target_id = item["object"]["id"]
        line = (
            f"EDGE {item['subject']['id']} ({item['subject']['label']}) "
            f"--[{item['predicate']}]--> {target_id} ({item['object']['label']})"
        )
        if item["qualifiers"]:
            qualifiers = "; ".join(
                f"{qualifier['key']}={format_value(qualifier['value'])}"
                for qualifier in item["qualifiers"]
            )
            line += f" [qualifiers: {qualifiers}]"
        lines.append(line)
    return "\n".join(lines)


def render_all(statements: list[dict[str, Any]], seed: int) -> dict[str, str]:
    return {
        "document": render_document(statements),
        "triple": render_triples(statements, seed),
        "graph": render_graph(statements),
    }


def build_bundle(
    train: list[dict[str, Any]],
    kb: dict[str, Any],
    limit: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    name_index = build_name_index(kb)
    candidates = [
        (index, sample)
        for index, sample in enumerate(train)
        if [step["function"] for step in sample.get("program", [])] == TARGET_PROGRAM
        and sample.get("answer") in sample.get("choices", [])
    ]
    random.Random(sample_seed).shuffle(candidates)

    bundle: list[dict[str, Any]] = []
    for train_index, sample in candidates:
        evidence = trace_two_hop(sample, kb, name_index)
        if evidence is None:
            continue
        contexts = render_all(evidence, seed=sample_seed + train_index)
        statement_ids = [item["statement_id"] for item in evidence]
        if len(statement_ids) != 2:
            raise RuntimeError("Expected exactly two gold statements")
        bundle.append(
            {
                "question_id": f"train_{train_index}",
                "train_index": train_index,
                "question": sample["question"],
                "choices": sample["choices"],
                "answer": sample["answer"],
                "program": sample["program"],
                "gold_statements": evidence,
                "gold_statement_ids": statement_ids,
                "contexts": contexts,
            }
        )
        if len(bundle) >= limit:
            break
    if len(bundle) != limit:
        raise RuntimeError(f"Only found {len(bundle)} executable questions; requested {limit}")
    return bundle


def make_prompt(item: dict[str, Any], representation: str) -> list[dict[str, str]]:
    user = f"""Knowledge representation: {representation}

Context:
{item['contexts'][representation]}

Question:
{item['question']}

Use only the context. Return only the exact answer entity or value, with no explanation."""
    return [
        {
            "role": "system",
            "content": "Answer the question using only the supplied gold evidence.",
        },
        {"role": "user", "content": user},
    ]


def call_chat_completions(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    retries: int,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body["choices"][0]["message"]
            text = (message.get("content") or "").strip()
            # Ollama exposes Qwen3 thinking through `reasoning`. With a small
            # generation budget it may finish before copying the final answer
            # into `content`, so retain the reasoning text for robust parsing.
            if not text:
                text = (message.get("reasoning") or "").strip()
            return text, body.get("usage", {})
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Model request failed: {last_error}")


def extract_choice(raw: str, choices: list[str]) -> str | None:
    stripped = raw.strip().strip('`').strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "answer" in parsed:
            stripped = str(parsed["answer"]).strip()
    except json.JSONDecodeError:
        pass

    letter_match = re.fullmatch(r"(?:answer\s*[:：]?\s*)?([A-J])(?:[.)])?", stripped, re.IGNORECASE)
    if letter_match:
        index = ord(letter_match.group(1).upper()) - ord("A")
        if index < len(choices):
            return choices[index]

    explicit_letters = re.findall(
        r"(?:correct\s+(?:choice|answer)(?:\s+should\s+be|\s+is)?|"
        r"(?:final\s+)?answer\s+is)\s*[:：]?\s*([A-J])\b",
        raw,
        re.IGNORECASE,
    )
    if explicit_letters:
        index = ord(explicit_letters[-1].upper()) - ord("A")
        if index < len(choices):
            return choices[index]

    # Qwen may retain a reasoning block. Prefer an explicit final standalone
    # option letter near the end before falling back to matching answer text.
    for line in reversed([line.strip() for line in raw.splitlines() if line.strip()]):
        line_match = re.fullmatch(r"(?:final answer\s*[:：]?\s*)?([A-J])(?:[.)])?", line, re.IGNORECASE)
        if line_match:
            index = ord(line_match.group(1).upper()) - ord("A")
            if index < len(choices):
                return choices[index]

    for choice in choices:
        if stripped.casefold() == choice.casefold():
            return choice
    matches = [choice for choice in choices if re.search(re.escape(choice), raw, re.IGNORECASE)]
    return matches[0] if len(matches) == 1 else None


def normalized_equal(left: str | None, right: str) -> bool:
    if left is None:
        return False
    normalize = lambda text: " ".join(text.casefold().split())
    return normalize(left) == normalize(right)


def evaluate(
    bundle: list[dict[str, Any]],
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    retries: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for number, item in enumerate(bundle, start=1):
        for representation in REPRESENTATIONS:
            raw, usage = call_chat_completions(
                base_url,
                api_key,
                model,
                make_prompt(item, representation),
                timeout,
                retries,
                max_tokens,
            )
            prediction = extract_choice(raw, item["choices"])
            predictions.append(
                {
                    "question_id": item["question_id"],
                    "representation": representation,
                    "gold_answer": item["answer"],
                    "prediction": prediction,
                    "raw_response": raw,
                    "correct": normalized_equal(prediction, item["answer"]),
                    "usage": usage,
                }
            )
        print(f"Completed {number}/{len(bundle)} questions", flush=True)

    by_representation: dict[str, dict[str, Any]] = {}
    for representation in REPRESENTATIONS:
        rows = [row for row in predictions if row["representation"] == representation]
        correct = sum(row["correct"] for row in rows)
        by_representation[representation] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows) if rows else None,
            "invalid_outputs": sum(row["prediction"] is None for row in rows),
        }

    all_correct = 0
    for item in bundle:
        rows = [row for row in predictions if row["question_id"] == item["question_id"]]
        all_correct += len(rows) == 3 and all(row["correct"] for row in rows)

    metrics = {
        "status": "completed",
        "model": model,
        "questions": len(bundle),
        "calls": len(predictions),
        "task_mode": "open_ended_exact_match_to_gold_choice",
        "chance_accuracy": None,
        "by_representation": by_representation,
        "all_three_correct": {
            "count": all_correct,
            "rate": all_correct / len(bundle),
        },
    }
    return predictions, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, default=WORKSPACE / "kb.json")
    parser.add_argument("--questions", type=Path, default=WORKSPACE / "train.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "oracle_20")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Qwen3 may spend part of this budget in its reasoning channel",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build and validate the Oracle bundle without calling a model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.kb.open(encoding="utf-8") as handle:
        kb = json.load(handle)
    with args.questions.open(encoding="utf-8") as handle:
        train = json.load(handle)

    bundle = build_bundle(train, kb, args.limit, args.sample_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "oracle_questions.jsonl", bundle)

    alignment = {
        "questions": len(bundle),
        "program_pattern": TARGET_PROGRAM,
        "statements_per_question": 2,
        "representations": list(REPRESENTATIONS),
        "same_question_set": True,
        "same_statement_ids_across_representations": True,
    }
    write_json(args.output_dir / "bundle_validation.json", alignment)

    if args.build_only or not (args.base_url and args.model):
        valid_paths = sum(
            len(item["gold_statement_ids"]) == 2
            and item["answer"] in item["choices"]
            and set(item["contexts"]) == set(REPRESENTATIONS)
            for item in bundle
        )
        metrics = {
            "status": "build_only",
            "reason": "No model call requested or OPENAI_BASE_URL/OPENAI_MODEL is missing",
            "questions": len(bundle),
            "calls": 0,
            "task_mode": "open_ended_exact_match_to_gold_choice",
            "chance_accuracy": None,
            "oracle_bundle_validation": {
                "correct": valid_paths,
                "total": len(bundle),
                "accuracy": valid_paths / len(bundle),
                "note": "Pipeline integrity metric; this is not LLM answer accuracy",
            },
            "llm_accuracy": None,
            "by_representation": {},
        }
        write_json(args.output_dir / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Bundle written to {args.output_dir.resolve()}")
        return

    predictions, metrics = evaluate(
        bundle,
        args.base_url,
        args.api_key,
        args.model,
        args.timeout,
        args.retries,
        args.max_tokens,
    )
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
