#!/usr/bin/env python3
"""D0 closed-book baseline for 2WikiMultihopQA.

This script samples 2WikiMultihopQA questions and optionally runs an
OpenAI-compatible chat model with question-only prompts. No context,
supporting facts, evidence triples, answer candidates, or retrieval results are
shown to the model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_DATA = PROJECT_ROOT / "2wikimultihop" / "data" / "data" / "dev.json"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(text: str) -> str:
    """2Wiki/Hotpot-style answer normalization."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def answer_f1(prediction: str, gold: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    zero = (0.0, 0.0, 0.0)

    if normalized_prediction in {"yes", "no", "noanswer"} and normalized_prediction != normalized_gold:
        return zero
    if normalized_gold in {"yes", "no", "noanswer"} and normalized_prediction != normalized_gold:
        return zero

    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return zero
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def closed_book_prompt(question: str, no_think: bool) -> list[dict[str, str]]:
    user_content = (
        f"Question:\n{question}\n\n"
        "Return only the exact answer entity, date, number, yes, or no. "
        "Do not include explanation."
    )
    if no_think:
        user_content = "/no_think\n\n" + user_content
    return [
        {
            "role": "system",
            "content": (
                "Answer the question from your own internal knowledge. "
                "No Wikipedia context, supporting facts, evidence triples, "
                "retrieval results, or answer candidates are provided."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def call_chat_completions(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    retries: int,
    max_tokens: int,
    ollama_think_false: bool,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if ollama_think_false:
        # Ollama's native chat API supports a top-level `think` flag for
        # thinking-capable models. Some OpenAI-compatible deployments also pass
        # it through; keep it opt-in because strict OpenAI-compatible servers
        # may reject unknown fields.
        payload["think"] = False
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            choice = result["choices"][0]
            content = choice["message"].get("content", "").strip()
            usage = result.get("usage", {})
            return content, usage, choice
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Chat completion failed after {retries + 1} attempts: {last_error}")


def select_questions(
    data: list[dict[str, Any]],
    question_types: list[str],
    per_type: int,
    seed: int,
    evidence_count: int | None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for question_type in question_types:
        candidates = [row for row in data if row.get("type") == question_type]
        if evidence_count is not None:
            candidates = [row for row in candidates if len(row.get("evidences", [])) == evidence_count]
        if len(candidates) < per_type:
            raise RuntimeError(
                f"Requested {per_type} {question_type!r} questions, but only "
                f"{len(candidates)} match the filters."
            )
        sampled = rng.sample(candidates, per_type)
        sampled.sort(key=lambda row: row["_id"])
        selected.extend(sampled)
    selected.sort(key=lambda row: (row["type"], row["_id"]))
    return selected


def build_question_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": row["_id"],
            "type": row["type"],
            "question": row["question"],
            "answer": row["answer"],
            "evidence_count": len(row.get("evidences", [])),
            "supporting_fact_count": len(row.get("supporting_facts", [])),
            "context_paragraph_count": len(row.get("context", [])),
        }
        for row in rows
    ]


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def usage_value(row: dict[str, Any], key: str) -> int:
    value = row.get("usage", {}).get(key)
    return int(value) if isinstance(value, (int, float)) else 0


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "total": 0,
            "answer_em": None,
            "answer_f1": None,
            "answer_precision": None,
            "answer_recall": None,
        }
    prompt_tokens = [usage_value(row, "prompt_tokens") for row in rows]
    completion_tokens = [usage_value(row, "completion_tokens") for row in rows]
    total_tokens = [usage_value(row, "total_tokens") for row in rows]
    exact = [1.0 if row["answer_em"] else 0.0 for row in rows]
    return {
        "total": len(rows),
        "answer_em": average(exact),
        "answer_f1": average([row["answer_f1"] for row in rows]),
        "answer_precision": average([row["answer_precision"] for row in rows]),
        "answer_recall": average([row["answer_recall"] for row in rows]),
        "empty_outputs": sum(not row["prediction"].strip() for row in rows),
        "empty_output_rate": sum(not row["prediction"].strip() for row in rows) / len(rows),
        "mean_prompt_tokens": average([float(value) for value in prompt_tokens]),
        "mean_completion_tokens": average([float(value) for value in completion_tokens]),
        "mean_total_tokens": average([float(value) for value in total_tokens]),
        "total_prompt_tokens": sum(prompt_tokens),
        "total_completion_tokens": sum(completion_tokens),
        "total_tokens": sum(total_tokens),
        "mean_model_latency_ms": average([row["model_latency_ms"] for row in rows]),
        "total_model_latency_ms": sum(row["model_latency_ms"] for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "d0_closed_book")
    parser.add_argument(
        "--types",
        nargs="+",
        default=["compositional", "inference"],
        help="Question types to sample independently.",
    )
    parser.add_argument("--per-type", type=int, default=50)
    parser.add_argument(
        "--evidence-count",
        type=int,
        default=2,
        help="Keep only questions with this many gold evidence triples. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Prefix prompts with /no_think for Qwen3-style thinking models.",
    )
    parser.add_argument(
        "--ollama-think-false",
        action="store_true",
        help="Send top-level think=false for Ollama-style thinking models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence_count = args.evidence_count if args.evidence_count > 0 else None
    data = read_json(args.data)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a list of examples in {args.data}")

    sampled = select_questions(data, args.types, args.per_type, args.seed, evidence_count)
    questions = build_question_rows(sampled)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "d0_questions.jsonl", questions)

    validation = {
        "condition": "D0_2wiki_closed_book",
        "source_data": str(args.data),
        "questions": len(questions),
        "sampling": {
            "types": args.types,
            "per_type": args.per_type,
            "seed": args.seed,
            "evidence_count_filter": evidence_count,
        },
        "by_type": {
            question_type: sum(row["type"] == question_type for row in questions)
            for question_type in args.types
        },
        "by_evidence_count": {
            str(count): sum(row["evidence_count"] == count for row in questions)
            for count in sorted({row["evidence_count"] for row in questions})
        },
        "knowledge_context_provided": False,
        "supporting_facts_loaded_into_prompt": False,
        "evidence_triples_loaded_into_prompt": False,
        "retrieval_used": False,
        "answer_candidates_shown_to_model": False,
    }
    write_json(args.output_dir / "bundle_validation.json", validation)

    if args.build_only or not (args.base_url and args.model):
        report = {"status": "build_only", **validation}
        write_json(args.output_dir / "metrics.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    predictions: list[dict[str, Any]] = []
    for number, item in enumerate(questions, start=1):
        start = time.perf_counter()
        raw, usage, choice = call_chat_completions(
            args.base_url,
            args.api_key,
            args.model,
            closed_book_prompt(item["question"], args.no_think),
            args.timeout,
            args.retries,
            args.max_tokens,
            args.ollama_think_false,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        f1, precision, recall = answer_f1(raw, item["answer"])
        message = choice.get("message", {})
        reasoning = message.get("reasoning", "")
        reasoning_content = message.get("reasoning_content", "")
        prediction = {
            **item,
            "gold_answer": item["answer"],
            "prediction": raw,
            "answer_em": exact_match(raw, item["answer"]),
            "answer_f1": f1,
            "answer_precision": precision,
            "answer_recall": recall,
            "usage": usage,
            "model_latency_ms": latency_ms,
            "response_diagnostics": {
                "finish_reason": choice.get("finish_reason"),
                "message_keys": sorted(message.keys()),
                "content_chars": len(message.get("content", "") or ""),
                "reasoning_chars": len(reasoning or ""),
                "reasoning_content_chars": len(reasoning_content or ""),
            },
        }
        predictions.append(prediction)
        print(f"Completed {number}/{len(questions)} ({item['type']})", flush=True)

    report = {
        "status": "completed",
        "condition": "D0_2wiki_closed_book",
        "model": args.model,
        "questions": len(questions),
        "calls": len(predictions),
        "configuration": {
            "knowledge_context_provided": False,
            "retrieval_used": False,
            "answer_candidates_shown_to_model": False,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "no_think": args.no_think,
            "ollama_think_false": args.ollama_think_false,
        },
        "sampling": validation["sampling"],
        "overall": summarise(predictions),
        "by_type": {
            question_type: summarise(
                [row for row in predictions if row["type"] == question_type]
            )
            for question_type in args.types
        },
        "by_evidence_count": {
            str(count): summarise(
                [row for row in predictions if row["evidence_count"] == count]
            )
            for count in sorted({row["evidence_count"] for row in predictions})
        },
    }
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "metrics.json", report)
    official_answer_predictions = {
        "answer": {row["question_id"]: row["prediction"] for row in predictions},
        "sp": {row["question_id"]: [] for row in predictions},
        "evidence": {row["question_id"]: [] for row in predictions},
    }
    write_json(args.output_dir / "official_prediction_format.json", official_answer_predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
