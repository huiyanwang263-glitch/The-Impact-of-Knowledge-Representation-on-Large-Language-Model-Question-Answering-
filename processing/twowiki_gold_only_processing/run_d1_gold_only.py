#!/usr/bin/env python3
"""D1 gold-only knowledge representation experiment for 2WikiMultihopQA.

For each selected 2WikiMultihopQA question, this script provides complete gold
evidence in one of three serializations:

- document: original gold supporting sentences
- triple: dataset-provided evidence triples
- graph: the same evidence triples serialized as directed graph edges

No distractor paragraphs, answer candidates, retrieval results, or non-gold
context are shown to the model.
"""

from __future__ import annotations

import argparse
import hashlib
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

REPRESENTATIONS = ("document", "triple", "graph")
EVIDENCE_ORDERS = ("dataset", "shuffled")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def stable_shuffle_seed(question_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


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
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if ollama_think_false:
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


def select_from_question_file(data: list[dict[str, Any]], question_file: Path) -> list[dict[str, Any]]:
    ids = [row["question_id"] for row in read_jsonl(question_file)]
    by_id = {row["_id"]: row for row in data}
    missing = [question_id for question_id in ids if question_id not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} question ids from {question_file} are missing in data")
    return [by_id[question_id] for question_id in ids]


def context_index(row: dict[str, Any]) -> dict[str, list[str]]:
    return {title: sentences for title, sentences in row.get("context", [])}


def gold_supporting_sentences(row: dict[str, Any]) -> list[dict[str, Any]]:
    by_title = context_index(row)
    sentences: list[dict[str, Any]] = []
    for title, sent_id in row.get("supporting_facts", []):
        title_sentences = by_title.get(title)
        text = None
        if title_sentences is not None and 0 <= int(sent_id) < len(title_sentences):
            text = title_sentences[int(sent_id)]
        sentences.append({"title": title, "sent_id": int(sent_id), "text": text})
    return sentences


def serialize_document(row: dict[str, Any]) -> str:
    lines = []
    for number, item in enumerate(gold_supporting_sentences(row), start=1):
        text = item["text"] if item["text"] is not None else "[MISSING SUPPORT SENTENCE]"
        lines.append(f"{number}. [{item['title']} #{item['sent_id']}] {text}")
    return "\n".join(lines)


def ordered_evidences(row: dict[str, Any], evidence_order: str, seed: int) -> list[list[str]]:
    evidences = [list(evidence) for evidence in row.get("evidences", [])]
    if evidence_order == "dataset":
        return evidences
    if evidence_order == "shuffled":
        rng = random.Random(stable_shuffle_seed(row["_id"], seed))
        rng.shuffle(evidences)
        return evidences
    raise ValueError(f"Unknown evidence order: {evidence_order}")


def serialize_triple(row: dict[str, Any], evidence_order: str, seed: int) -> str:
    lines = []
    for number, evidence in enumerate(ordered_evidences(row, evidence_order, seed), start=1):
        subject, relation, obj = evidence
        lines.append(f"{number}. ({subject} | {relation} | {obj})")
    return "\n".join(lines)


def serialize_graph(row: dict[str, Any], evidence_order: str, seed: int) -> str:
    lines = []
    for number, evidence in enumerate(ordered_evidences(row, evidence_order, seed), start=1):
        subject, relation, obj = evidence
        lines.append(f"{number}. EDGE {subject} --[{relation}]--> {obj}")
    return "\n".join(lines)


def serialize_evidence(row: dict[str, Any], representation: str, evidence_order: str, seed: int) -> str:
    if representation == "document":
        return serialize_document(row)
    if representation == "triple":
        return serialize_triple(row, evidence_order, seed)
    if representation == "graph":
        return serialize_graph(row, evidence_order, seed)
    raise ValueError(f"Unknown representation: {representation}")


def gold_only_prompt(question: str, evidence_text: str, representation: str, no_think: bool) -> list[dict[str, str]]:
    user_content = (
        f"Question:\n{question}\n\n"
        f"Gold evidence representation: {representation}\n"
        f"Gold evidence:\n{evidence_text}\n\n"
        "Use only the gold evidence above to answer the question. "
        "Return only the exact answer entity, date, number, yes, or no. "
        "Do not include explanation."
    )
    if no_think:
        user_content = "/no_think\n\n" + user_content
    return [
        {
            "role": "system",
            "content": (
                "You answer 2WikiMultihopQA questions using complete gold evidence. "
                "No distractor context, retrieval results, or answer candidates are provided."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def build_items(
    rows: list[dict[str, Any]],
    representations: list[str],
    evidence_order: str,
    seed: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        support = gold_supporting_sentences(row)
        missing_support = sum(item["text"] is None for item in support)
        for representation in representations:
            items.append(
                {
                    "question_id": row["_id"],
                    "type": row["type"],
                    "representation": representation,
                    "question": row["question"],
                    "answer": row["answer"],
                    "evidence_count": len(row.get("evidences", [])),
                    "supporting_fact_count": len(row.get("supporting_facts", [])),
                    "missing_supporting_sentences": missing_support,
                    "evidence_order": evidence_order if representation in {"triple", "graph"} else "supporting_fact_order",
                    "evidence_text": serialize_evidence(row, representation, evidence_order, seed),
                    "gold_evidences": row.get("evidences", []),
                    "gold_supporting_facts": row.get("supporting_facts", []),
                }
            )
    return items


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
    parser.add_argument("--questions-from", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "d1_gold_only")
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    parser.add_argument(
        "--evidence-order",
        choices=EVIDENCE_ORDERS,
        default="dataset",
        help=(
            "Order used for triple and graph evidence. Use 'shuffled' to match "
            "the KQA Pro unordered-triple control more closely."
        ),
    )
    parser.add_argument("--types", nargs="+", default=["compositional", "inference"])
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
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--ollama-think-false", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence_count = args.evidence_count if args.evidence_count > 0 else None
    data = read_json(args.data)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a list of examples in {args.data}")

    if args.questions_from:
        selected = select_from_question_file(data, args.questions_from)
        sampling = {
            "mode": "questions_from",
            "questions_from": str(args.questions_from),
            "question_count": len(selected),
        }
    else:
        selected = select_questions(data, args.types, args.per_type, args.seed, evidence_count)
        sampling = {
            "mode": "stratified_sample",
            "types": args.types,
            "per_type": args.per_type,
            "seed": args.seed,
            "evidence_count_filter": evidence_count,
        }

    items = build_items(selected, args.representations, args.evidence_order, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "d1_items.jsonl", items)

    validation = {
        "condition": "D1_2wiki_gold_only",
        "source_data": str(args.data),
        "questions": len(selected),
        "calls": len(items),
        "representations": args.representations,
        "evidence_order": args.evidence_order,
        "sampling": sampling,
        "by_type": {
            question_type: sum(row.get("type") == question_type for row in selected)
            for question_type in sorted({row.get("type") for row in selected})
        },
        "by_representation": {
            representation: sum(item["representation"] == representation for item in items)
            for representation in args.representations
        },
        "knowledge_context_provided": True,
        "gold_only": True,
        "distractor_context_provided": False,
        "retrieval_used": False,
        "answer_candidates_shown_to_model": False,
        "missing_supporting_sentences": sum(item["missing_supporting_sentences"] for item in items),
    }
    write_json(args.output_dir / "bundle_validation.json", validation)

    if args.build_only or not (args.base_url and args.model):
        report = {"status": "build_only", **validation}
        write_json(args.output_dir / "metrics.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    predictions: list[dict[str, Any]] = []
    for number, item in enumerate(items, start=1):
        start = time.perf_counter()
        raw, usage, choice = call_chat_completions(
            args.base_url,
            args.api_key,
            args.model,
            gold_only_prompt(
                item["question"],
                item["evidence_text"],
                item["representation"],
                args.no_think,
            ),
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
            "question_id": item["question_id"],
            "type": item["type"],
            "representation": item["representation"],
            "question": item["question"],
            "gold_answer": item["answer"],
            "prediction": raw,
            "answer_em": exact_match(raw, item["answer"]),
            "answer_f1": f1,
            "answer_precision": precision,
            "answer_recall": recall,
            "evidence_count": item["evidence_count"],
            "supporting_fact_count": item["supporting_fact_count"],
            "missing_supporting_sentences": item["missing_supporting_sentences"],
            "evidence_text": item["evidence_text"],
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
        print(
            f"Completed {number}/{len(items)} "
            f"({item['type']}, {item['representation']})",
            flush=True,
        )

    report = {
        "status": "completed",
        "condition": "D1_2wiki_gold_only",
        "model": args.model,
        "questions": len(selected),
        "calls": len(predictions),
        "configuration": {
            "knowledge_context_provided": True,
            "gold_only": True,
            "distractor_context_provided": False,
            "retrieval_used": False,
            "answer_candidates_shown_to_model": False,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "no_think": args.no_think,
            "ollama_think_false": args.ollama_think_false,
        },
        "sampling": sampling,
        "overall": summarise(predictions),
        "by_representation": {
            representation: summarise(
                [row for row in predictions if row["representation"] == representation]
            )
            for representation in args.representations
        },
        "by_type": {
            question_type: summarise(
                [row for row in predictions if row["type"] == question_type]
            )
            for question_type in sorted({row["type"] for row in predictions})
        },
        "by_representation_and_type": {
            representation: {
                question_type: summarise(
                    [
                        row
                        for row in predictions
                        if row["representation"] == representation and row["type"] == question_type
                    ]
                )
                for question_type in sorted({row["type"] for row in predictions})
            }
            for representation in args.representations
        },
    }
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
