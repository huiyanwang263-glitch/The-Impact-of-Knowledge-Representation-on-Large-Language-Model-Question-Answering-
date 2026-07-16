#!/usr/bin/env python3
"""KQA Pro E3-M: LLM evidence selection over a manual candidate bank.

This experiment reuses the fixed 15-question Oracle bundle (five linear
questions for each of 1/2/3 hops). For every evaluation question it builds a
bounded candidate bank containing the complete gold path plus true statements
materialized from other, non-evaluation KQA Pro questions. The same candidate
statement IDs and order are rendered as Document, Triple, and Graph.

For each representation an LLM selector first returns ranked statement IDs.
Only those selected statements are then sent to the answer model. Gold is
present in the candidate bank but is never identified or forced into the E3
answer context.

This is bounded LLM evidence selection, not full-KB RAG. The script never reads
the generated full-KB/atomic RAG corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PROCESSING = HERE.parent
PROJECT = PROCESSING.parent
ORACLE_DIR = PROCESSING / "knowledge_representation_processing"
sys.path.insert(0, str(ORACLE_DIR))

import run_oracle_20 as oracle_base  # noqa: E402
import run_oracle_hops_15 as oracle_hops  # noqa: E402
from build_pilot_representations import format_value, statement_to_sentence  # noqa: E402


REPS = ("document", "triple", "graph")
EXPECTED_HOP_COUNTS = {1: 5, 2: 5, 3: 5}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def stable_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_evaluation_questions(questions: list[dict[str, Any]]) -> None:
    if len(questions) != 15:
        raise RuntimeError(f"Expected the shared 15-question bundle, found {len(questions)}")
    hop_counts = Counter(int(item["hop_count"]) for item in questions)
    if dict(hop_counts) != EXPECTED_HOP_COUNTS:
        raise RuntimeError(
            f"Expected five questions for each of 1/2/3 hops, found {dict(hop_counts)}"
        )
    ids = [str(item["question_id"]) for item in questions]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Evaluation question IDs are not unique")
    for item in questions:
        gold_ids = item.get("gold_statement_ids", [])
        gold = item.get("gold_statements", [])
        if len(gold_ids) != int(item["hop_count"]) or len(gold) != len(gold_ids):
            raise RuntimeError(
                f"{item['question_id']}: hop count and gold statement count disagree"
            )
        if gold_ids != [statement["statement_id"] for statement in gold]:
            raise RuntimeError(f"{item['question_id']}: gold IDs and statements disagree")


def build_question_derived_noise_bank(
    train: list[dict[str, Any]],
    kb: dict[str, Any],
    evaluation_questions: list[dict[str, Any]],
    background_per_hop: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Materialize true facts from non-evaluation question paths."""
    excluded_indices = {int(item["train_index"]) for item in evaluation_questions}
    name_index = oracle_base.build_name_index(kb)
    statements: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[str]] = defaultdict(list)

    for hop in sorted(EXPECTED_HOP_COUNTS):
        candidates = [
            (index, sample)
            for index, sample in enumerate(train)
            if index not in excluded_indices
            and oracle_hops.is_linear_program(sample, hop)
            and sample.get("answer") in sample.get("choices", [])
        ]
        random.Random(seed + hop).shuffle(candidates)
        accepted = 0
        for train_index, sample in candidates:
            path = oracle_hops.trace_path(sample, hop, kb, name_index)
            if path is None:
                continue
            source_id = f"train_{train_index}"
            for statement in path:
                statement_id = statement["statement_id"]
                statements[statement_id] = statement
                if len(sources[statement_id]) < 20:
                    sources[statement_id].append(source_id)
            accepted += 1
            if accepted == background_per_hop:
                break
        if accepted != background_per_hop:
            raise RuntimeError(
                f"Found {accepted} background {hop}-hop paths; "
                f"requested {background_per_hop}"
            )

    return [
        {
            "statement_id": statement_id,
            "statement": statements[statement_id],
            "source_question_ids": sorted(set(sources[statement_id])),
        }
        for statement_id in sorted(statements)
    ]


def entity_ids(statement: dict[str, Any]) -> set[str]:
    result = {statement["subject"]["id"]}
    if statement["object"]["kind"] != "literal":
        result.add(statement["object"]["id"])
    return result


def mentions_choices(statement: dict[str, Any], choices: list[str]) -> bool:
    protected = {" ".join(str(choice).casefold().split()) for choice in choices}
    labels = {
        " ".join(str(statement["subject"]["label"]).casefold().split()),
        " ".join(str(statement["object"]["label"]).casefold().split()),
    }
    return bool(protected & labels)


def noise_indexes(
    noise_bank: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, list[str]], list[str]]:
    statements: dict[str, dict[str, Any]] = {}
    by_predicate: dict[str, list[str]] = defaultdict(list)
    by_entity: dict[str, list[str]] = defaultdict(list)
    for row in noise_bank:
        statement = row["statement"]
        statement_id = statement["statement_id"]
        statements[statement_id] = statement
        by_predicate[statement["predicate"]].append(statement_id)
        for entity_id in entity_ids(statement):
            by_entity[entity_id].append(statement_id)
    return statements, dict(by_predicate), dict(by_entity), sorted(statements)


def choose_noise(
    question: dict[str, Any],
    statements: dict[str, dict[str, Any]],
    by_predicate: dict[str, list[str]],
    by_entity: dict[str, list[str]],
    all_ids: list[str],
    needed: int,
    seed: int,
) -> list[dict[str, Any]]:
    gold = question["gold_statements"]
    excluded = set(question["gold_statement_ids"])
    predicates = {statement["predicate"] for statement in gold}
    entities = {entity_id for statement in gold for entity_id in entity_ids(statement)}

    same_predicate = [
        statement_id
        for predicate in sorted(predicates)
        for statement_id in by_predicate.get(predicate, [])
    ]
    shared_entity = [
        statement_id
        for entity_id in sorted(entities)
        for statement_id in by_entity.get(entity_id, [])
    ]
    fallback = list(all_ids)
    rng = random.Random(seed + int(question["train_index"]) * 1009)
    rng.shuffle(same_predicate)
    rng.shuffle(shared_entity)
    rng.shuffle(fallback)

    selected: list[str] = []
    selected_set = set(excluded)
    for statement_id in same_predicate + shared_entity + fallback:
        if statement_id in selected_set:
            continue
        statement = statements[statement_id]
        # Avoid extra mentions of the correct answer or distractor choices.
        # Gold statements remain untouched and can naturally mention the answer.
        if mentions_choices(statement, question.get("choices", [])):
            continue
        selected.append(statement_id)
        selected_set.add(statement_id)
        if len(selected) == needed:
            break
    if len(selected) != needed:
        raise RuntimeError(
            f"{question['question_id']}: found {len(selected)} usable noise statements; "
            f"needed {needed}"
        )
    return [statements[statement_id] for statement_id in selected]


def qualifier_suffix(statement: dict[str, Any]) -> str:
    qualifiers = statement.get("qualifiers", [])
    if not qualifiers:
        return ""
    rendered = "; ".join(
        f"{item['key']}={format_value(item['value'])}" for item in qualifiers
    )
    return f" [qualifiers: {rendered}]"


def render_statement(statement: dict[str, Any], representation: str) -> str:
    statement_id = statement["statement_id"]
    subject = statement["subject"]
    obj = statement["object"]
    if representation == "document":
        text = statement_to_sentence(statement)
    elif representation == "triple":
        text = (
            f"({subject['label']} | {statement['predicate']} | {obj['label']})"
            + qualifier_suffix(statement)
        )
    elif representation == "graph":
        target_id = (
            f"literal_{statement_id}" if obj["kind"] == "literal" else obj["id"]
        )
        text = (
            f"EDGE {subject['id']} ({subject['label']}) "
            f"--[{statement['predicate']}]--> {target_id} ({obj['label']})"
            + qualifier_suffix(statement)
        )
    else:
        raise ValueError(representation)
    return f"[{statement_id}] {text}"


def render_statements(statements: list[dict[str, Any]], representation: str) -> str:
    # The input order is reused verbatim for all representations.
    return "\n".join(render_statement(statement, representation) for statement in statements)


def build_candidate_banks(
    questions: list[dict[str, Any]],
    noise_bank: list[dict[str, Any]],
    candidate_statements: int,
    seed: int,
) -> list[dict[str, Any]]:
    statements, by_predicate, by_entity, all_ids = noise_indexes(noise_bank)
    result: list[dict[str, Any]] = []
    for question in questions:
        gold = list(question["gold_statements"])
        if len(gold) >= candidate_statements:
            raise RuntimeError(
                f"{question['question_id']}: gold count must be below candidate count"
            )
        noise = choose_noise(
            question,
            statements,
            by_predicate,
            by_entity,
            all_ids,
            candidate_statements - len(gold),
            seed,
        )
        candidates = gold + noise
        random.Random(seed + int(question["train_index"]) * 9173).shuffle(candidates)
        candidate_ids = [statement["statement_id"] for statement in candidates]
        result.append(
            {
                "question_id": question["question_id"],
                "train_index": question["train_index"],
                "hop_count": question["hop_count"],
                "question": question["question"],
                "choices": question["choices"],
                "answer": question["answer"],
                "gold_statement_ids": question["gold_statement_ids"],
                "gold_statements": gold,
                "noise_statement_ids": [statement["statement_id"] for statement in noise],
                "candidate_statement_ids": candidate_ids,
                "candidate_statements": candidates,
                "candidate_count": len(candidates),
                "candidate_order_shared_across_representations": True,
                "candidate_contexts": {
                    rep: render_statements(candidates, rep) for rep in REPS
                },
            }
        )
    return result


def selector_prompt(
    item: dict[str, Any], representation: str, top_k: int
) -> list[dict[str, str]]:
    user = f"""/no_think

Knowledge representation: {representation}

Candidate statements:
{item['candidate_contexts'][representation]}

Question:
{item['question']}

Select up to {top_k} statements most useful for answering the question.
Do not answer the question. Use only IDs shown in square brackets.
Return exactly one JSON object in this form:
{{"selected_statement_ids": ["stmt_id_1", "stmt_id_2"]}}
List one to {top_k} unique IDs from most to least useful. Never repeat an ID,
and do not add irrelevant statements merely to reach {top_k}."""
    return [
        {
            "role": "system",
            "content": (
                "You are an evidence selector. Rank candidate evidence using the question, "
                "and output only valid JSON containing candidate statement IDs."
            ),
        },
        {"role": "user", "content": user},
    ]


def native_ollama_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    if root.endswith("/api/chat"):
        return root
    return root + "/api/chat"


def call_ollama_structured_selector(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    top_k: int,
    timeout: int,
    retries: int,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "selected_statement_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": top_k,
            }
        },
        "required": ["selected_statement_ids"],
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
    endpoint = native_ollama_url(base_url)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            usage = {
                "prompt_tokens": body.get("prompt_eval_count", 0),
                "completion_tokens": body.get("eval_count", 0),
                "total_tokens": body.get("prompt_eval_count", 0)
                + body.get("eval_count", 0),
            }
            return (body.get("message", {}).get("content") or "").strip(), usage
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Structured selector request failed: {last_error}")


def json_candidates(raw: str) -> Iterable[Any]:
    stripped = raw.strip().strip("`").strip()
    try:
        yield json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{.*?\}", raw, re.DOTALL):
        try:
            yield json.loads(match.group(0))
        except json.JSONDecodeError:
            continue


def parse_selection(
    raw: str, valid_ids: set[str], top_k: int
) -> tuple[list[str], list[str], int]:
    values: list[Any] | None = None
    for parsed in json_candidates(raw):
        if isinstance(parsed, dict) and isinstance(parsed.get("selected_statement_ids"), list):
            values = parsed["selected_statement_ids"]
            break
        if isinstance(parsed, list):
            values = parsed
            break
    if values is None:
        return [], [], 0

    selected: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for value in values:
        statement_id = str(value).strip()
        if not statement_id or statement_id in seen:
            continue
        seen.add(statement_id)
        if statement_id not in valid_ids:
            unknown.append(statement_id)
            continue
        selected.append(statement_id)
        if len(selected) == top_k:
            break
    return selected, unknown, len(values)


def retrieval_metrics(selected_ids: list[str], gold_ids: list[str]) -> dict[str, Any]:
    gold = set(gold_ids)
    covered = gold & set(selected_ids)
    first_gold_rank = next(
        (rank for rank, statement_id in enumerate(selected_ids, 1) if statement_id in gold),
        None,
    )
    return {
        "gold_statement_count": len(gold),
        "selected_statement_count": len(selected_ids),
        "covered_gold_statement_ids": sorted(covered),
        "statement_recall_at_k": len(covered) / len(gold),
        "hit_at_k": bool(covered),
        "complete_path_recall_at_k": gold <= set(selected_ids),
        "first_gold_rank": first_gold_rank,
        "reciprocal_rank": 1 / first_gold_rank if first_gold_rank else 0.0,
    }


def answer_prompt(question: str, context: str, representation: str) -> list[dict[str, str]]:
    user = f"""Knowledge representation: {representation}

Selected context:
{context}

Question:
{question}

Use only the selected context. Return only the exact answer entity or value, with no explanation."""
    return [
        {
            "role": "system",
            "content": "Answer the question using only the evidence selected in the previous stage.",
        },
        {"role": "user", "content": user},
    ]


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]], answer_ran: bool) -> dict[str, Any]:
    selected_counts = [
        row["retrieval_metrics"]["selected_statement_count"] for row in rows
    ]
    result: dict[str, Any] = {
        "questions": len(rows),
        "selector_valid_rate": average(
            [float(row["selector_valid"]) for row in rows]
        ),
        "mean_statement_recall_at_k": average(
            [row["retrieval_metrics"]["statement_recall_at_k"] for row in rows]
        ),
        "hit_rate_at_k": average(
            [float(row["retrieval_metrics"]["hit_at_k"]) for row in rows]
        ),
        "complete_path_recall_at_k": average(
            [
                float(row["retrieval_metrics"]["complete_path_recall_at_k"])
                for row in rows
            ]
        ),
        "retrieval_mrr": average(
            [row["retrieval_metrics"]["reciprocal_rank"] for row in rows]
        ),
        "mean_selected_statement_count": average(
            [float(value) for value in selected_counts]
        ),
        "min_selected_statement_count": min(selected_counts) if selected_counts else None,
        "max_selected_statement_count": max(selected_counts) if selected_counts else None,
        "mean_selector_latency_seconds": average(
            [row["selector_latency_seconds"] for row in rows]
        ),
        "mean_selector_prompt_tokens": average(
            [float(row["selector_usage"].get("prompt_tokens", 0)) for row in rows]
        ),
    }
    if answer_ran:
        answered = [row for row in rows if row.get("answer_attempted")]
        correct = sum(bool(row.get("correct")) for row in rows)
        invalid = sum(row.get("prediction") is None for row in rows)
        result["answer"] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows) if rows else None,
            "answer_attempted": len(answered),
            "invalid_or_unattempted": invalid,
        }
        result["answer_efficiency"] = {
            "mean_latency_seconds": average(
                [row["answer_latency_seconds"] for row in answered]
            ),
            "mean_prompt_tokens": average(
                [float(row["answer_usage"].get("prompt_tokens", 0)) for row in answered]
            ),
            "mean_completion_tokens": average(
                [float(row["answer_usage"].get("completion_tokens", 0)) for row in answered]
            ),
            "mean_total_tokens": average(
                [float(row["answer_usage"].get("total_tokens", 0)) for row in answered]
            ),
        }
    return result


def build_report(
    rows: list[dict[str, Any]], config: dict[str, Any], answer_ran: bool
) -> dict[str, Any]:
    return {
        "status": "completed" if answer_ran else "selection_only",
        "condition": "E3_manual_candidate_bank_llm_selection",
        "questions": len({row["question_id"] for row in rows}),
        "selector_runs": len(rows),
        "answer_calls": sum(bool(row.get("answer_attempted")) for row in rows),
        "configuration": config,
        "by_representation": {
            rep: summarize([row for row in rows if row["representation"] == rep], answer_ran)
            for rep in REPS
        },
        "by_hop": {
            str(hop): {
                rep: summarize(
                    [
                        row
                        for row in rows
                        if row["hop_count"] == hop and row["representation"] == rep
                    ],
                    answer_ran,
                )
                for rep in REPS
            }
            for hop in sorted(EXPECTED_HOP_COUNTS)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ORACLE_DIR / "output" / "oracle_hops_15" / "oracle_questions.jsonl",
    )
    parser.add_argument("--train", type=Path, default=PROJECT / "train.json")
    parser.add_argument("--kb", type=Path, default=PROJECT / "kb.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "e3_manual_hops_15")
    parser.add_argument("--background-per-hop", type=int, default=50)
    parser.add_argument("--candidate-statements", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--representations", nargs="+", choices=REPS, default=list(REPS))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--selector-format-retries", type=int, default=1)
    parser.add_argument("--selector-max-tokens", type=int, default=512)
    parser.add_argument(
        "--selector-api",
        choices=("ollama-native", "openai-compatible"),
        default="ollama-native",
    )
    parser.add_argument("--answer-max-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.questions.exists():
        raise FileNotFoundError(
            f"Shared Oracle question bundle not found: {args.questions}. "
            "Build E1 oracle_hops_15 first; E3 must not independently resample questions."
        )
    if args.top_k >= args.candidate_statements:
        raise ValueError("--top-k must be smaller than --candidate-statements")
    if args.top_k < max(EXPECTED_HOP_COUNTS):
        raise ValueError("--top-k must be large enough to contain the longest gold path")

    questions = list(read_jsonl(args.questions))
    validate_evaluation_questions(questions)
    with args.train.open(encoding="utf-8") as handle:
        train = json.load(handle)
    with args.kb.open(encoding="utf-8") as handle:
        kb = json.load(handle)

    noise_bank = build_question_derived_noise_bank(
        train,
        kb,
        questions,
        args.background_per_hop,
        args.noise_seed,
    )
    candidates = build_candidate_banks(
        questions,
        noise_bank,
        args.candidate_statements,
        args.noise_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "question_derived_noise_bank.jsonl", noise_bank)
    write_jsonl(args.output_dir / "candidate_banks.jsonl", candidates)

    candidate_hash = stable_digest(
        [
            {"question_id": item["question_id"], "ids": item["candidate_statement_ids"]}
            for item in candidates
        ]
    )
    config = {
        "condition": "E3-M bounded manual candidate bank with LLM selection",
        "questions": 15,
        "questions_per_hop": 5,
        "hops": [1, 2, 3],
        "candidate_statements": args.candidate_statements,
        "selected_top_k": args.top_k,
        "selection_policy": "one_to_top_k_unique_relevant_statements",
        "background_questions_per_hop": args.background_per_hop,
        "noise_source": "gold paths from non-evaluation KQA Pro questions",
        "full_kb_rag_used": False,
        "gold_present_in_candidate_bank": True,
        "gold_forced_into_e3_selected_context": False,
        "candidate_ids_and_order_shared_across_representations": True,
        "candidate_bundle_sha256": candidate_hash,
        "question_bundle_sha256": stable_digest(
            [
                {
                    "question_id": item["question_id"],
                    "gold_statement_ids": item["gold_statement_ids"],
                }
                for item in questions
            ]
        ),
        "noise_bank_sha256": stable_digest(
            [row["statement_id"] for row in noise_bank]
        ),
        "noise_seed": args.noise_seed,
        "representations": args.representations,
        "selector_format_retries": args.selector_format_retries,
        "selector_thinking": False,
        "selector_api": args.selector_api,
        "model": args.model or None,
        "selector_temperature": 0,
        "answer_temperature": 0,
    }
    write_json(args.output_dir / "configuration.json", config)
    validation = {
        "questions": len(candidates),
        "hop_counts": dict(Counter(str(item["hop_count"]) for item in candidates)),
        "all_candidate_counts_match": all(
            item["candidate_count"] == args.candidate_statements for item in candidates
        ),
        "all_gold_complete_in_candidate_bank": all(
            set(item["gold_statement_ids"]) <= set(item["candidate_statement_ids"])
            for item in candidates
        ),
        "gold_noise_disjoint": all(
            not (set(item["gold_statement_ids"]) & set(item["noise_statement_ids"]))
            for item in candidates
        ),
        "all_candidate_ids_unique": all(
            len(item["candidate_statement_ids"])
            == len(set(item["candidate_statement_ids"]))
            for item in candidates
        ),
        "candidate_order_shared_across_representations": True,
    }
    if not all(
        value is True
        for key, value in validation.items()
        if key not in {"questions", "hop_counts"}
    ):
        raise RuntimeError(f"Candidate bundle validation failed: {validation}")
    write_json(args.output_dir / "bundle_validation.json", validation)

    model_configured = bool(args.base_url and args.model)
    if args.build_only or not model_configured:
        write_json(
            args.output_dir / "metrics.json",
            {
                "status": "build_only",
                "condition": "E3_manual_candidate_bank_llm_selection",
                "llm_selection_ran": False,
                "llm_answer_ran": False,
                "configuration": config,
                "validation": validation,
            },
        )
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return

    run_signature = stable_digest(
        {
            "candidate_bundle_sha256": candidate_hash,
            "model": args.model,
            "representations": args.representations,
            "top_k": args.top_k,
            "selection_policy": "one_to_top_k_unique_relevant_statements",
            "selection_only": args.selection_only,
            "selector_max_tokens": args.selector_max_tokens,
            "selector_api": args.selector_api,
            "answer_max_tokens": args.answer_max_tokens,
        }
    )
    checkpoint_path = args.output_dir / "e3_results_checkpoint.jsonl"
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if checkpoint_path.exists():
        for saved in read_jsonl(checkpoint_path):
            if saved.get("run_signature") == run_signature:
                completed[(str(saved["question_id"]), saved["representation"])] = saved

    for question_number, item in enumerate(candidates, 1):
        statement_by_id = {
            statement["statement_id"]: statement
            for statement in item["candidate_statements"]
        }
        valid_ids = set(statement_by_id)
        for representation in args.representations:
            result_key = (str(item["question_id"]), representation)
            if result_key in completed:
                continue
            selector_raw_attempts: list[str] = []
            selector_usage_attempts: list[dict[str, Any]] = []
            selector_latency = 0.0
            selected_ids: list[str] = []
            unknown_ids: list[str] = []
            raw_selection_count = 0
            for _ in range(args.selector_format_retries + 1):
                started = time.perf_counter()
                if args.selector_api == "ollama-native":
                    raw, usage = call_ollama_structured_selector(
                        args.base_url,
                        args.model,
                        selector_prompt(item, representation, args.top_k),
                        args.top_k,
                        args.timeout,
                        args.retries,
                        args.selector_max_tokens,
                    )
                else:
                    raw, usage = oracle_base.call_chat_completions(
                        args.base_url,
                        args.api_key,
                        args.model,
                        selector_prompt(item, representation, args.top_k),
                        args.timeout,
                        args.retries,
                        args.selector_max_tokens,
                    )
                selector_latency += time.perf_counter() - started
                selector_raw_attempts.append(raw)
                selector_usage_attempts.append(usage)
                selected_ids, unknown_ids, raw_selection_count = parse_selection(
                    raw, valid_ids, args.top_k
                )
                if selected_ids:
                    break

            selector_valid = bool(selected_ids)
            selected_statements = [statement_by_id[sid] for sid in selected_ids]
            context = render_statements(selected_statements, representation)
            row: dict[str, Any] = {
                "run_signature": run_signature,
                "question_id": item["question_id"],
                "train_index": item["train_index"],
                "hop_count": item["hop_count"],
                "representation": representation,
                "question": item["question"],
                "gold_answer": item["answer"],
                "gold_statement_ids": item["gold_statement_ids"],
                "candidate_statement_ids": item["candidate_statement_ids"],
                "selected_statement_ids": selected_ids,
                "selected_statements": selected_statements,
                "selected_context": context,
                "selector_valid": selector_valid,
                "selector_attempts": len(selector_raw_attempts),
                "selector_raw_selection_count": raw_selection_count,
                "selector_unknown_ids": unknown_ids,
                "selector_raw_responses": selector_raw_attempts,
                "selector_usage_attempts": selector_usage_attempts,
                "selector_usage": selector_usage_attempts[-1],
                "selector_latency_seconds": selector_latency,
                "retrieval_metrics": retrieval_metrics(
                    selected_ids, item["gold_statement_ids"]
                ),
                "answer_attempted": False,
                "prediction": None,
                "correct": False,
            }
            if selector_valid and not args.selection_only:
                started = time.perf_counter()
                answer_raw, answer_usage = oracle_base.call_chat_completions(
                    args.base_url,
                    args.api_key,
                    args.model,
                    answer_prompt(item["question"], context, representation),
                    args.timeout,
                    args.retries,
                    args.answer_max_tokens,
                )
                row["answer_latency_seconds"] = time.perf_counter() - started
                row["answer_attempted"] = True
                row["answer_raw_response"] = answer_raw
                row["answer_usage"] = answer_usage
                row["prediction"] = oracle_base.extract_choice(
                    answer_raw, item["choices"]
                )
                row["correct"] = oracle_base.normalized_equal(
                    row["prediction"], item["answer"]
                )
            append_jsonl(checkpoint_path, row)
            completed[result_key] = row
        print(f"Completed {question_number}/{len(candidates)} questions", flush=True)

    question_order = {
        str(item["question_id"]): index for index, item in enumerate(candidates)
    }
    representation_order = {rep: index for index, rep in enumerate(REPS)}
    rows = sorted(
        completed.values(),
        key=lambda row: (
            question_order[str(row["question_id"])],
            representation_order[row["representation"]],
        ),
    )
    write_jsonl(args.output_dir / "selection_results.jsonl", rows)
    if not args.selection_only:
        write_jsonl(args.output_dir / "predictions.jsonl", rows)
    report = build_report(rows, config, answer_ran=not args.selection_only)
    write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
