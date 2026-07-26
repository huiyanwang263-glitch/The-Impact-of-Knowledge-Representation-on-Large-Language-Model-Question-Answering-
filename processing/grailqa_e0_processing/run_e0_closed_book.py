#!/usr/bin/env python3
"""Run GrailQA E0 with question-only prompts and answer-set evaluation."""

from __future__ import annotations

import argparse
import json
import os
import string
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DEFAULT_QUESTIONS = (
    HERE.parent
    / "grailqa_mini_kb_processing"
    / "output"
    / "dev_10_per_hop"
    / "selected_questions.jsonl"
)
DEFAULT_ALIAS_CACHE = HERE / "output" / "answer_alias_cache.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = value.replace("–", "-").replace("—", "-")
    value = "".join(" " if char in string.punctuation else char for char in value)
    tokens = value.split()
    if len(tokens) > 1:
        tokens = [token for token in tokens if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def normalized_set(values: Iterable[str]) -> set[str]:
    return {normalized for value in values if (normalized := normalize_answer(value))}


def score_canonical_answer_set(predicted: list[str], gold: list[str]) -> dict[str, Any]:
    predicted_set = normalized_set(predicted)
    gold_set = normalized_set(gold)
    overlap = predicted_set & gold_set
    precision = len(overlap) / len(predicted_set) if predicted_set else 0.0
    recall = len(overlap) / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "canonical_answer_set_em": predicted_set == gold_set,
        "canonical_answer_precision": precision,
        "canonical_answer_recall": recall,
        "canonical_answer_f1": f1,
        "normalized_prediction_set": sorted(predicted_set),
        "normalized_canonical_gold_set": sorted(gold_set),
    }


def load_alias_cache(path: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        raise RuntimeError(f"Alias cache has no entities object: {path}")
    aliases: dict[str, set[str]] = {}
    for mid, row in entities.items():
        values = row.get("accepted_normalized_aliases", [])
        normalized_values = {str(value) for value in values if str(value)}
        for canonical_name in row.get("canonical_names", []):
            if normalized := normalize_answer(str(canonical_name)):
                normalized_values.add(normalized)
        for alias in row.get("aliases", []):
            for surface_form in alias.get("surface_forms", []):
                if normalized := normalize_answer(str(surface_form)):
                    normalized_values.add(normalized)
        aliases[str(mid)] = normalized_values
    metadata = {
        "path": str(path.resolve()),
        "cache_sha256": payload.get("cache_sha256"),
        "schema_version": payload.get("schema_version"),
        "target_answer_mid_count": payload.get("target_answer_mid_count"),
        "normalized_alias_count": payload.get("audit", {}).get(
            "normalized_alias_count"
        ),
        "evaluation_normalization_version": 2,
        "raw_surface_forms_renormalized": True,
    }
    return aliases, metadata


def build_alias_gold_groups(
    gold: list[str],
    gold_ids: list[str],
    alias_cache: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Group indistinguishable Gold MIDs by their normalized answer label."""
    if len(gold) != len(gold_ids):
        raise ValueError(
            "gold answers and gold IDs must have equal lengths: "
            f"{len(gold)} != {len(gold_ids)}"
        )
    groups: dict[str, dict[str, Any]] = {}
    for answer, raw_mid in zip(gold, gold_ids):
        mid = str(raw_mid)
        normalized = normalize_answer(answer)
        if not normalized:
            continue
        group_key = normalized
        group = groups.setdefault(
            group_key,
            {
                "group_key": group_key,
                "canonical_answers": [],
                "gold_mids": [],
                "accepted_normalized_aliases": set(),
            },
        )
        if answer not in group["canonical_answers"]:
            group["canonical_answers"].append(answer)
        if mid not in group["gold_mids"]:
            group["gold_mids"].append(mid)
        if normalized:
            group["accepted_normalized_aliases"].add(normalized)
        group["accepted_normalized_aliases"].update(alias_cache.get(mid, set()))
    return list(groups.values())


def maximum_alias_group_matching(
    predicted: list[str],
    gold_groups: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Return maximum one-to-one matches as (prediction index, Gold group index)."""
    normalized_predictions = [normalize_answer(value) for value in predicted]
    adjacency = [
        [
            group_index
            for group_index, group in enumerate(gold_groups)
            if normalized
            and normalized in group["accepted_normalized_aliases"]
        ]
        for normalized in normalized_predictions
    ]
    group_to_prediction: dict[int, int] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for group_index in adjacency[prediction_index]:
            if group_index in visited:
                continue
            visited.add(group_index)
            previous = group_to_prediction.get(group_index)
            if previous is None or augment(previous, visited):
                group_to_prediction[group_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predicted)):
        augment(prediction_index, set())
    return sorted(
        (
            (prediction_index, group_index)
            for group_index, prediction_index in group_to_prediction.items()
        ),
        key=lambda pair: pair[0],
    )


def score_alias_answer_set(
    predicted: list[str],
    gold: list[str],
    gold_ids: list[str],
    alias_cache: dict[str, set[str]],
) -> dict[str, Any]:
    """Score aliases against answer-label groups rather than raw Freebase MIDs."""
    unique_predictions: list[str] = []
    seen: set[str] = set()
    for value in predicted:
        normalized = normalize_answer(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_predictions.append(value)
    gold_groups = build_alias_gold_groups(gold, gold_ids, alias_cache)
    matches = maximum_alias_group_matching(unique_predictions, gold_groups)
    matched_prediction_indexes = {index for index, _ in matches}
    matched_group_indexes = {group_index for _, group_index in matches}
    matched_count = len(matches)
    precision = matched_count / len(unique_predictions) if unique_predictions else 0.0
    recall = matched_count / len(gold_groups) if gold_groups else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "alias_scoring_version": 2,
        "alias_scoring_unit": "normalized_gold_answer_group",
        "alias_answer_set_em": (
            matched_count == len(gold_groups) == len(unique_predictions)
        ),
        "alias_answer_precision": precision,
        "alias_answer_recall": recall,
        "alias_answer_f1": f1,
        "alias_matches": [
            {
                "prediction": unique_predictions[index],
                "normalized_prediction": normalize_answer(unique_predictions[index]),
                "gold_group": gold_groups[group_index]["group_key"],
                "gold_mids": gold_groups[group_index]["gold_mids"],
            }
            for index, group_index in matches
        ],
        "unmatched_predictions": [
            value
            for index, value in enumerate(unique_predictions)
            if index not in matched_prediction_indexes
        ],
        "alias_gold_group_count": len(gold_groups),
        "alias_gold_groups": [
            {
                "group_key": group["group_key"],
                "canonical_answers": group["canonical_answers"],
                "gold_mids": group["gold_mids"],
            }
            for group in gold_groups
        ],
        "unmatched_gold_groups": [
            {
                "group_key": group["group_key"],
                "canonical_answers": group["canonical_answers"],
                "gold_mids": group["gold_mids"],
            }
            for group_index, group in enumerate(gold_groups)
            if group_index not in matched_group_indexes
        ],
        "unmatched_gold_ids": [
            mid
            for group_index, group in enumerate(gold_groups)
            if group_index not in matched_group_indexes
            for mid in group["gold_mids"]
        ],
    }


def evaluation_metadata(alias_cache_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "canonical_unit": "normalized_answer_string_set",
        "alias_aware_unit": "normalized_gold_answer_group",
        "gold_mids_with_same_normalized_answer_are_grouped": True,
        "normalization_version": 2,
        "single_token_articles_are_preserved": True,
        "alias_cache": alias_cache_metadata,
    }


def select_questions(
    rows: list[dict[str, Any]], max_per_hop: int | None
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_hop: Counter[int] = Counter()
    for row in sorted(rows, key=lambda item: (item["hop_count"], item["dataset_index"])):
        hop = int(row["hop_count"])
        if max_per_hop is not None and per_hop[hop] >= max_per_hop:
            continue
        gold_names = [
            answer.get("entity_name") or answer["answer_argument"]
            for answer in row["answers"]
        ]
        selected.append(
            {
                "question_id": str(row["question_id"]),
                "dataset_index": int(row["dataset_index"]),
                "hop_count": hop,
                "level": row.get("level"),
                "question": row["question"],
                "gold_answers": gold_names,
                "gold_answer_ids": list(row["answer_ids"]),
            }
        )
        per_hop[hop] += 1
    return selected


def closed_book_prompt(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer from your own internal knowledge. No knowledge base, Gold path, "
                "SPARQL query, logical form, retrieved evidence, or answer candidates are "
                "provided. A question may have one or multiple correct answers."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                "Return the complete answer set as JSON. Use the exact format "
                '{"answers":["answer 1","answer 2"]}. Include no explanation.'
            ),
        },
    ]


def native_ollama_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    if root.endswith("/api/chat"):
        return root
    return root + "/api/chat"


def call_ollama(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    retries: int,
    max_tokens: int,
) -> tuple[str, dict[str, int]]:
    schema = {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type": "string", "maxLength": 160},
                "minItems": 1,
                "maxItems": 128,
            }
        },
        "required": ["answers"],
    }
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
    ).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                native_ollama_url(base_url),
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            raw = (body.get("message", {}).get("content") or "").strip()
            prompt_tokens = int(body.get("prompt_eval_count", 0) or 0)
            completion_tokens = int(body.get("eval_count", 0) or 0)
            return raw, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Ollama request failed after {retries + 1} attempts: {last_error}")


def parse_answers(raw: str) -> tuple[list[str], bool, str | None]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], False, repr(exc)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answers"), list):
        return [], False, "Expected an object containing an answers array"
    answers = [str(value).strip() for value in parsed["answers"] if str(value).strip()]
    if not answers:
        return [], False, "The answers array was empty"
    return answers, True, None


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_score_group(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    return {
        "correct_sets": sum(bool(row[f"{prefix}_answer_set_em"]) for row in rows),
        "answer_set_em": average(
            [float(row[f"{prefix}_answer_set_em"]) for row in rows]
        ),
        "answer_precision": average(
            [row[f"{prefix}_answer_precision"] for row in rows]
        ),
        "answer_recall": average([row[f"{prefix}_answer_recall"] for row in rows]),
        "answer_f1": average([row[f"{prefix}_answer_f1"] for row in rows]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"questions": 0, "canonical": None, "alias_aware": None}
    return {
        "questions": len(rows),
        "canonical": summarize_score_group(rows, "canonical"),
        "alias_aware": summarize_score_group(rows, "alias"),
        "valid_json_rate": average([float(row["output_valid"]) for row in rows]),
        "mean_latency_seconds": average([row["latency_seconds"] for row in rows]),
        "mean_prompt_tokens": average(
            [float(row["usage"]["prompt_tokens"]) for row in rows]
        ),
        "mean_completion_tokens": average(
            [float(row["usage"]["completion_tokens"]) for row in rows]
        ),
        "mean_total_tokens": average(
            [float(row["usage"]["total_tokens"]) for row in rows]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--alias-cache", type=Path, default=DEFAULT_ALIAS_CACHE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "e0_30")
    parser.add_argument("--max-per-hop", type=int)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3-4b-16k"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.questions.exists():
        raise FileNotFoundError(args.questions)
    if not args.alias_cache.exists():
        raise FileNotFoundError(
            f"Alias cache not found: {args.alias_cache}. Run build_answer_alias_cache.py first."
        )
    if args.max_per_hop is not None and args.max_per_hop < 1:
        raise ValueError("--max-per-hop must be positive")
    questions = select_questions(read_jsonl(args.questions), args.max_per_hop)
    alias_cache, alias_cache_metadata = load_alias_cache(args.alias_cache)
    if not questions:
        raise RuntimeError("No E0 questions were selected")
    required_gold_ids = {
        mid for question in questions for mid in question["gold_answer_ids"]
    }
    missing_alias_ids = sorted(required_gold_ids - set(alias_cache))
    if missing_alias_ids:
        raise RuntimeError(
            "Alias cache is missing evaluation Gold MIDs: "
            + ", ".join(missing_alias_ids[:20])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "e0_questions.jsonl", questions)
    hops = sorted({row["hop_count"] for row in questions})
    validation = {
        "condition": "E0_grailqa_closed_book",
        "source_questions": str(args.questions.resolve()),
        "questions": len(questions),
        "by_hop": {
            str(hop): sum(row["hop_count"] == hop for row in questions)
            for hop in hops
        },
        "knowledge_context_provided": False,
        "mini_kb_loaded_into_prompt": False,
        "gold_path_loaded_into_prompt": False,
        "sparql_loaded_into_prompt": False,
        "logical_form_loaded_into_prompt": False,
        "retrieval_used": False,
        "answer_candidates_shown_to_model": False,
        "alias_cache": alias_cache_metadata,
        "alias_cache_loaded_into_prompt": False,
        "alias_evaluation_only": True,
        "required_gold_mid_count": len(required_gold_ids),
        "missing_alias_gold_mids": missing_alias_ids,
    }
    write_json(args.output_dir / "bundle_validation.json", validation)
    if args.build_only:
        report = {"status": "build_only", **validation}
        write_json(args.output_dir / "metrics.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    predictions: list[dict[str, Any]] = []
    for number, item in enumerate(questions, start=1):
        started = time.perf_counter()
        raw, usage = call_ollama(
            args.base_url,
            args.model,
            closed_book_prompt(item["question"]),
            args.timeout,
            args.retries,
            args.max_tokens,
        )
        latency = time.perf_counter() - started
        predicted, valid, parse_error = parse_answers(raw)
        canonical_scores = score_canonical_answer_set(
            predicted, item["gold_answers"]
        )
        alias_scores = score_alias_answer_set(
            predicted,
            item["gold_answers"],
            item["gold_answer_ids"],
            alias_cache,
        )
        predictions.append(
            {
                **item,
                "prediction": predicted,
                "raw_response": raw,
                "output_valid": valid,
                "parse_error": parse_error,
                **canonical_scores,
                **alias_scores,
                "usage": usage,
                "latency_seconds": latency,
            }
        )
        write_jsonl(args.output_dir / "predictions.jsonl", predictions)
        print(f"Completed {number}/{len(questions)} ({item['hop_count']}-hop)", flush=True)

    report = {
        "status": "completed",
        "metrics_schema_version": 2,
        "condition": "E0_grailqa_closed_book",
        "model": args.model,
        "questions": len(questions),
        "calls": len(predictions),
        "evaluation": evaluation_metadata(alias_cache_metadata),
        "configuration": {
            "knowledge_context_provided": False,
            "retrieval_used": False,
            "answer_candidates_shown_to_model": False,
            "structured_json": True,
            "think": False,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "maximum_answer_string_characters": 160,
            "maximum_answer_count": 128,
            "max_per_hop": args.max_per_hop,
            "alias_cache": alias_cache_metadata,
            "alias_cache_loaded_into_prompt": False,
        },
        "overall": summarize(predictions),
        "by_hop": {
            str(hop): summarize(
                [row for row in predictions if row["hop_count"] == hop]
            )
            for hop in hops
        },
    }
    write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
