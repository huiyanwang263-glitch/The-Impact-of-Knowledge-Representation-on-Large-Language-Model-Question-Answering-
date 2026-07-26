#!/usr/bin/env python3
"""Run GrailQA E3 over fixed true-noise banks without full-Freebase RAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PROCESSING = HERE.parent
PROJECT = HERE.parent.parent
E0_DIR = PROCESSING / "grailqa_e0_processing"
GOLD_HELPER_DIR = PROCESSING / "grailqa_gold_path_processing"
sys.path.insert(0, str(E0_DIR))
sys.path.insert(0, str(GOLD_HELPER_DIR))
import run_e0_closed_book as e0  # noqa: E402
import build_gold_paths as gold  # noqa: E402


MINI_OUTPUT = PROCESSING / "grailqa_mini_kb_processing" / "output" / "dev_10_per_hop"
GRAILQA_DATA = PROJECT / "GrailQA_v1.0 (1)" / "GrailQA_v1.0"
DEFAULT_QUESTIONS = MINI_OUTPUT / "selected_questions.jsonl"
DEFAULT_KB = MINI_OUTPUT / "mini_kb.jsonl"
DEFAULT_QUERY_RESULTS = MINI_OUTPUT / "query_results.jsonl"
DEFAULT_DATASETS = [
    GRAILQA_DATA / "grailqa_v1.0_train.json",
    GRAILQA_DATA / "grailqa_v1.0_dev.json",
]
DEFAULT_ALIAS_CACHE = E0_DIR / "output" / "answer_alias_cache.json"
REPRESENTATIONS = ("document", "triple", "graph")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def serialized_statement(statement: dict[str, Any]) -> dict[str, Any]:
    subject = statement["subject"]
    obj = statement["object"]
    subject_text = subject.get("label") or subject["id"]
    object_text = obj.get("label") or obj["id"]
    relation = statement["predicate"]
    relation_text = statement.get("predicate_label") or relation
    return {
        **statement,
        "document": f"{subject_text} has relation {relation_text} with {object_text}.",
        "triple": f"({subject_text}, {relation_text}, {object_text})",
        "graph": f"{subject_text} --[{relation}]--> {object_text}",
    }


def node_from_graph(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(node["id"]),
        "label": str(node.get("friendly_name") or node["id"]),
        "kind": "entity" if node.get("node_type") == "entity" else "literal",
    }


def offline_noise_bank(
    dataset_paths: list[Path], evaluation_qids: set[str], evaluation_gold_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    statements: dict[str, dict[str, Any]] = {}
    source_qids: dict[str, set[str]] = defaultdict(set)
    source_hops: dict[str, set[int]] = defaultdict(set)
    audit: Counter[str] = Counter()
    for path in dataset_paths:
        rows = read_json(path)
        if not isinstance(rows, list):
            raise RuntimeError(f"Expected a JSON array in {path}")
        audit["dataset_rows"] += len(rows)
        for row in rows:
            qid = str(row.get("qid"))
            if qid in evaluation_qids:
                audit["excluded_evaluation_question"] += 1
                continue
            if row.get("function") != "none" or not row.get("answer"):
                audit["excluded_function_or_empty_answer"] += 1
                continue
            if any(answer.get("answer_type") != "Entity" for answer in row["answer"]):
                audit["excluded_non_entity_answer"] += 1
                continue
            graph_query = row.get("graph_query", {})
            nodes = graph_query.get("nodes", [])
            edges = graph_query.get("edges", [])
            class_nodes = [node for node in nodes if node.get("node_type") == "class"]
            if len(class_nodes) != 1 or class_nodes[0].get("question_node") != 1:
                audit["excluded_requires_unobserved_intermediate_binding"] += 1
                continue
            qnode_id = int(class_nodes[0]["nid"])
            fixed_nodes = {
                int(node["nid"]): node_from_graph(node)
                for node in nodes
                if int(node["nid"]) != qnode_id
            }
            hop = int(row.get("num_edge", len(edges)))
            for answer in row["answer"]:
                binding = dict(fixed_nodes)
                binding[qnode_id] = {
                    "id": str(answer["answer_argument"]),
                    "label": str(answer.get("entity_name") or answer["answer_argument"]),
                    "kind": "entity",
                }
                for edge in edges:
                    subject = binding[int(edge["start"])]
                    obj = binding[int(edge["end"])]
                    identity = {
                        "subject_id": subject["id"],
                        "predicate": edge["relation"],
                        "object_id": obj["id"],
                    }
                    statement_id = gold.stable_id("gstmt", identity)
                    if statement_id in evaluation_gold_ids:
                        audit["excluded_evaluation_gold_overlap"] += 1
                        continue
                    statement = serialized_statement(
                        {
                            "statement_id": statement_id,
                            "kind": "relation",
                            "subject": subject,
                            "predicate": edge["relation"],
                            "predicate_label": edge.get("friendly_name") or edge["relation"],
                            "object": obj,
                            "qualifiers": [],
                        }
                    )
                    statements.setdefault(statement_id, statement)
                    source_qids[statement_id].add(qid)
                    source_hops[statement_id].add(hop)
            audit["accepted_background_questions"] += 1

    result = []
    for statement_id in sorted(statements):
        result.append(
            {
                **statements[statement_id],
                "source_question_ids": sorted(source_qids[statement_id]),
                "source_hops": sorted(source_hops[statement_id]),
            }
        )
    audit["deduplicated_noise_statements"] = len(result)
    return result, dict(audit)


def latest_successful_bindings(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") == "success":
            latest[str(row["question_id"])] = row
    return latest


def labels_from_kb(kb_rows: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for statement in kb_rows:
        for side in ("subject", "object"):
            node = statement[side]
            labels[str(node["id"])] = str(node.get("label") or node["id"])
    return labels


def sufficient_gold(
    question: dict[str, Any],
    query_result: dict[str, Any],
    kb_by_id: dict[str, dict[str, Any]],
    labels: dict[str, str],
) -> tuple[list[str], dict[str, list[str]]]:
    question_nodes = [
        node
        for node in question["graph_query"]["nodes"]
        if node.get("question_node") == 1
    ]
    if len(question_nodes) != 1:
        raise RuntimeError(f"Expected one question node for {question['question_id']}")
    question_variable = f"x{int(question_nodes[0]['nid'])}"
    by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in query_result["bindings"]:
        if question_variable in binding:
            by_answer[str(binding[question_variable]["id"])].append(binding)

    ordered_ids: list[str] = []
    paths: dict[str, list[str]] = {}
    seen: set[str] = set()
    for answer_id in question["answer_ids"]:
        alternatives = by_answer.get(str(answer_id), [])
        if not alternatives:
            raise RuntimeError(
                f"No grounded binding for answer {answer_id} in {question['question_id']}"
            )
        alternatives.sort(
            key=lambda binding: compact_json(
                {name: value["id"] for name, value in binding.items()}
            )
        )
        binding = alternatives[0]
        path_ids: list[str] = []
        for edge in question["graph_query"]["edges"]:
            statement = gold.build_statement(edge, binding, labels)
            statement_id = statement["statement_id"]
            if statement_id not in kb_by_id:
                raise RuntimeError(
                    f"Sufficient Gold statement {statement_id} missing from mini-KB"
                )
            path_ids.append(statement_id)
            if statement_id not in seen:
                seen.add(statement_id)
                ordered_ids.append(statement_id)
        paths[str(answer_id)] = path_ids
    return ordered_ids, paths


def candidate_banks(
    questions: list[dict[str, Any]],
    kb_by_id: dict[str, dict[str, Any]],
    query_results: dict[str, dict[str, Any]],
    noise_rows: list[dict[str, Any]],
    noise_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    labels = labels_from_kb(list(kb_by_id.values()))
    noise_by_id = {row["statement_id"]: row for row in noise_rows}
    sorted_noise_ids = sorted(noise_by_id)
    banks: list[dict[str, Any]] = []
    for question in questions:
        qid = str(question["question_id"])
        if qid not in query_results:
            raise RuntimeError(f"No successful materialization checkpoint for {qid}")
        gold_ids, gold_paths = sufficient_gold(
            question, query_results[qid], kb_by_id, labels
        )
        persisted_gold_ids = question.get("sufficient_gold_statement_ids")
        if persisted_gold_ids is not None and list(persisted_gold_ids) != gold_ids:
            raise RuntimeError(
                f"Persisted sufficient Gold differs from recomputed Gold for {qid}"
            )
        persisted_paths = question.get("sufficient_gold_paths_by_answer")
        if persisted_paths is not None and dict(persisted_paths) != gold_paths:
            raise RuntimeError(
                f"Persisted sufficient Gold paths differ from recomputed paths for {qid}"
            )
        gold_set = set(gold_ids)
        answer_ids = set(str(mid) for mid in question["answer_ids"])
        eligible_noise = [
            statement_id
            for statement_id in sorted_noise_ids
            if statement_id not in gold_set
            and str(noise_by_id[statement_id]["subject"]["id"]) not in answer_ids
            and str(noise_by_id[statement_id]["object"]["id"]) not in answer_ids
        ]
        if len(eligible_noise) < noise_count:
            raise RuntimeError(
                f"Question {qid} has only {len(eligible_noise)} eligible noise facts; "
                f"requested {noise_count}"
            )
        question_seed = int(
            hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).hexdigest()[:16], 16
        )
        rng = random.Random(question_seed)
        sampled_noise_ids = rng.sample(eligible_noise, noise_count)
        candidate_ids = gold_ids + sampled_noise_ids
        rng.shuffle(candidate_ids)
        statements = []
        for statement_id in candidate_ids:
            statement = kb_by_id.get(statement_id) or noise_by_id[statement_id]
            statements.append(statement)
        banks.append(
            {
                "question_id": qid,
                "dataset_index": int(question["dataset_index"]),
                "hop_count": int(question["hop_count"]),
                "level": question.get("level"),
                "question": question["question"],
                "gold_answers": [
                    answer.get("entity_name") or answer["answer_argument"]
                    for answer in question["answers"]
                ],
                "gold_answer_ids": list(question["answer_ids"]),
                "sufficient_gold_statement_ids": gold_ids,
                "sufficient_gold_paths_by_answer": gold_paths,
                "full_materialized_gold_statement_count": len(
                    question["gold_statement_ids"]
                ),
                "noise_statement_ids": sampled_noise_ids,
                "candidate_statement_ids": candidate_ids,
                "candidate_statements": statements,
            }
        )
    return banks


def select_questions(
    rows: list[dict[str, Any]], max_per_hop: int | None
) -> list[dict[str, Any]]:
    if max_per_hop is None:
        return list(rows)
    selected: list[dict[str, Any]] = []
    per_hop: Counter[int] = Counter()
    for row in sorted(rows, key=lambda item: (item["hop_count"], item["dataset_index"])):
        hop = int(row["hop_count"])
        if per_hop[hop] >= max_per_hop:
            continue
        selected.append(row)
        per_hop[hop] += 1
    return selected


def render_candidates(bank: dict[str, Any], representation: str) -> str:
    return "\n".join(
        f"[{statement['statement_id']}] {statement[representation]}"
        for statement in bank["candidate_statements"]
    )


def selector_prompt(question: str, candidates: str, top_k: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Select and rank only the candidate statements needed to answer the "
                "question. Return candidate statement IDs only in the required JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\nCandidate statements:\n{candidates}\n\n"
                f"Return between 1 and {top_k} unique relevant IDs as "
                '{"selected_statement_ids":["gstmt_..."]}. Do not add facts.'
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


def call_selector(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    top_k: int,
    timeout: int,
    retries: int,
    max_tokens: int,
) -> tuple[str, dict[str, int]]:
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
            prompt_tokens = int(body.get("prompt_eval_count", 0) or 0)
            completion_tokens = int(body.get("eval_count", 0) or 0)
            return (body.get("message", {}).get("content") or "").strip(), {
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
    raise RuntimeError(f"Selector request failed after {retries + 1} attempts: {last_error}")


def parse_selection(
    raw: str, valid_ids: set[str], top_k: int
) -> tuple[list[str], list[str], bool, str | None]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], [], False, repr(exc)
    values = parsed.get("selected_statement_ids") if isinstance(parsed, dict) else None
    if not isinstance(values, list):
        return [], [], False, "Expected selected_statement_ids array"
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
        elif len(selected) < top_k:
            selected.append(statement_id)
    valid = bool(selected) and not unknown
    return selected, unknown, valid, None if valid else "Empty selection or unknown IDs"


def retrieval_metrics(selected: list[str], gold_ids: list[str]) -> dict[str, Any]:
    gold_set = set(gold_ids)
    covered = gold_set & set(selected)
    first_rank = next(
        (rank for rank, statement_id in enumerate(selected, 1) if statement_id in gold_set),
        None,
    )
    return {
        "gold_statement_count": len(gold_set),
        "selected_statement_count": len(selected),
        "covered_gold_statement_ids": sorted(covered),
        "statement_recall_at_k": len(covered) / len(gold_set),
        "hit_at_k": bool(covered),
        "complete_sufficient_gold_recall_at_k": gold_set <= set(selected),
        "first_gold_rank": first_rank,
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
    }


def answer_prompt(question: str, context: str, representation: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer using only the selected evidence. Return final answer entities, "
                "not intermediate entities or descriptions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Knowledge representation: {representation}\n\n"
                f"Selected evidence:\n{context}\n\nQuestion:\n{question}\n\n"
                "Return the complete final answer set as JSON in the exact format "
                '{"answers":["answer 1","answer 2"]}. Include no explanation.'
            ),
        },
    ]


def selected_context(
    selected_ids: list[str], bank_by_id: dict[str, dict[str, Any]], representation: str
) -> str:
    return "\n".join(
        f"[{statement_id}] {bank_by_id[statement_id][representation]}"
        for statement_id in selected_ids
    )


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]], answer_ran: bool) -> dict[str, Any]:
    selected_counts = [
        row["retrieval_metrics"]["selected_statement_count"] for row in rows
    ]
    retrieval = {
        "questions": len(rows),
        "selector_valid_rate": average([float(row["selector_valid"]) for row in rows]),
        "mean_statement_recall_at_k": average(
            [row["retrieval_metrics"]["statement_recall_at_k"] for row in rows]
        ),
        "hit_rate_at_k": average(
            [float(row["retrieval_metrics"]["hit_at_k"]) for row in rows]
        ),
        "complete_sufficient_gold_recall_at_k": average(
            [
                float(row["retrieval_metrics"]["complete_sufficient_gold_recall_at_k"])
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
        "mean_selector_completion_tokens": average(
            [float(row["selector_usage"].get("completion_tokens", 0)) for row in rows]
        ),
        "mean_selector_total_tokens": average(
            [float(row["selector_usage"].get("total_tokens", 0)) for row in rows]
        ),
    }
    if answer_ran:
        retrieval["answer"] = e0.summarize(rows)
    return retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB)
    parser.add_argument("--query-results", type=Path, default=DEFAULT_QUERY_RESULTS)
    parser.add_argument("--datasets", type=Path, nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--alias-cache", type=Path, default=DEFAULT_ALIAS_CACHE)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "output" / "e3_manual_30"
    )
    parser.add_argument("--noise-statements", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-hop", type=int)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument(
        "--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3-4b-16k"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--selector-max-tokens", type=int, default=1024)
    parser.add_argument("--answer-max-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.questions, args.kb, args.query_results, args.alias_cache, *args.datasets]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.noise_statements < 1 or args.top_k < 1:
        raise ValueError("--noise-statements and --top-k must be positive")
    if args.max_per_hop is not None and args.max_per_hop < 1:
        raise ValueError("--max-per-hop must be positive")

    all_questions = read_jsonl(args.questions)
    questions = select_questions(all_questions, args.max_per_hop)
    kb_rows = read_jsonl(args.kb)
    kb_by_id = {row["statement_id"]: row for row in kb_rows}
    if len(kb_by_id) != len(kb_rows):
        raise RuntimeError("mini-KB contains duplicate statement IDs")
    evaluation_qids = {str(row["question_id"]) for row in all_questions}
    evaluation_gold_ids = {
        statement_id
        for row in all_questions
        for statement_id in row["gold_statement_ids"]
    }
    noise_rows, noise_audit = offline_noise_bank(
        args.datasets, evaluation_qids, evaluation_gold_ids
    )
    query_results = latest_successful_bindings(args.query_results)
    banks = candidate_banks(
        questions,
        kb_by_id,
        query_results,
        noise_rows,
        args.noise_statements,
        args.seed,
    )
    used_noise_ids = {
        statement_id for bank in banks for statement_id in bank["noise_statement_ids"]
    }
    used_noise_rows = [
        row for row in noise_rows if row["statement_id"] in used_noise_ids
    ]
    alias_cache, alias_metadata = e0.load_alias_cache(args.alias_cache)
    missing_alias_ids = sorted(
        {mid for bank in banks for mid in bank["gold_answer_ids"]} - set(alias_cache)
    )
    if missing_alias_ids:
        raise RuntimeError(f"Alias cache misses Gold MIDs: {missing_alias_ids[:20]}")

    max_gold = max(len(bank["sufficient_gold_statement_ids"]) for bank in banks)
    if max_gold > args.top_k:
        raise RuntimeError(
            f"Top-K {args.top_k} is smaller than maximum sufficient Gold size {max_gold}"
        )
    candidate_hash = sha256_value(
        [
            {
                "question_id": bank["question_id"],
                "gold": bank["sufficient_gold_statement_ids"],
                "candidates": bank["candidate_statement_ids"],
            }
            for bank in banks
        ]
    )
    max_candidate_chars = {
        representation: max(len(render_candidates(bank, representation)) for bank in banks)
        for representation in REPRESENTATIONS
    }
    validation = {
        "condition": "E3_grailqa_manual_llm_retrieval",
        "questions": len(banks),
        "by_hop": {
            str(hop): sum(bank["hop_count"] == hop for bank in banks)
            for hop in sorted({bank["hop_count"] for bank in banks})
        },
        "representations": list(REPRESENTATIONS),
        "full_freebase_retrieval_used": False,
        "noise_source": "offline-grounded Gold edges from non-evaluation GrailQA train/dev questions",
        "noise_audit": noise_audit,
        "noise_statements_per_question": args.noise_statements,
        "deduplicated_background_noise_pool_count": len(noise_rows),
        "persisted_sampled_noise_union_count": len(used_noise_rows),
        "candidate_statement_count_min": min(len(bank["candidate_statement_ids"]) for bank in banks),
        "candidate_statement_count_max": max(len(bank["candidate_statement_ids"]) for bank in banks),
        "sufficient_gold_statement_count_min": min(len(bank["sufficient_gold_statement_ids"]) for bank in banks),
        "sufficient_gold_statement_count_max": max_gold,
        "selector_top_k": args.top_k,
        "complete_gold_theoretically_retrievable_at_k": max_gold <= args.top_k,
        "gold_present_in_every_candidate_bank": all(
            set(bank["sufficient_gold_statement_ids"])
            <= set(bank["candidate_statement_ids"])
            for bank in banks
        ),
        "gold_noise_disjoint": all(
            set(bank["sufficient_gold_statement_ids"]).isdisjoint(bank["noise_statement_ids"])
            for bank in banks
        ),
        "candidate_ids_unique": all(
            len(bank["candidate_statement_ids"]) == len(set(bank["candidate_statement_ids"]))
            for bank in banks
        ),
        "candidate_ids_and_order_shared_across_representations": True,
        "gold_forced_into_selector_output": False,
        "max_candidate_context_characters": max_candidate_chars,
        "candidate_bundle_sha256": candidate_hash,
        "alias_cache": alias_metadata,
        "alias_cache_loaded_into_prompt": False,
        "missing_alias_gold_mids": missing_alias_ids,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "question_derived_noise_bank.jsonl", used_noise_rows
    )
    write_jsonl(args.output_dir / "candidate_banks.jsonl", banks)
    write_json(args.output_dir / "bundle_validation.json", validation)
    if args.build_only:
        report = {"status": "build_only", **validation}
        write_json(args.output_dir / "metrics.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    run_signature = sha256_value(
        {
            "candidate_bundle_sha256": candidate_hash,
            "model": args.model,
            "top_k": args.top_k,
            "max_per_hop": args.max_per_hop,
            "selector_max_tokens": args.selector_max_tokens,
            "answer_max_tokens": args.answer_max_tokens,
            "selection_only": args.selection_only,
        }
    )
    checkpoint_path = args.output_dir / "e3_checkpoint.jsonl"
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(checkpoint_path):
        if row.get("run_signature") == run_signature:
            cached[(str(row["question_id"]), str(row["representation"]))] = row

    results: list[dict[str, Any]] = []
    total = len(banks) * len(REPRESENTATIONS)
    completed = 0
    for bank in banks:
        bank_by_id = {
            statement["statement_id"]: statement
            for statement in bank["candidate_statements"]
        }
        for representation in REPRESENTATIONS:
            completed += 1
            cache_key = (bank["question_id"], representation)
            if cache_key in cached:
                results.append(cached[cache_key])
                continue
            started = time.perf_counter()
            raw_selector, selector_usage = call_selector(
                args.base_url,
                args.model,
                selector_prompt(
                    bank["question"], render_candidates(bank, representation), args.top_k
                ),
                args.top_k,
                args.timeout,
                args.retries,
                args.selector_max_tokens,
            )
            selector_latency = time.perf_counter() - started
            selected_ids, unknown_ids, selector_valid, selector_error = parse_selection(
                raw_selector, set(bank["candidate_statement_ids"]), args.top_k
            )
            row: dict[str, Any] = {
                "run_signature": run_signature,
                "question_id": bank["question_id"],
                "dataset_index": bank["dataset_index"],
                "hop_count": bank["hop_count"],
                "level": bank["level"],
                "representation": representation,
                "question": bank["question"],
                "gold_answers": bank["gold_answers"],
                "gold_answer_ids": bank["gold_answer_ids"],
                "sufficient_gold_statement_ids": bank["sufficient_gold_statement_ids"],
                "selected_statement_ids": selected_ids,
                "unknown_statement_ids": unknown_ids,
                "selector_valid": selector_valid,
                "selector_error": selector_error,
                "raw_selector_response": raw_selector,
                "selector_usage": selector_usage,
                "selector_latency_seconds": selector_latency,
                "retrieval_metrics": retrieval_metrics(
                    selected_ids, bank["sufficient_gold_statement_ids"]
                ),
                "answer_attempted": False,
                "prediction": [],
                "raw_response": "",
                "output_valid": False,
                "parse_error": None,
                **e0.score_canonical_answer_set([], bank["gold_answers"]),
                **e0.score_alias_answer_set(
                    [],
                    bank["gold_answers"],
                    bank["gold_answer_ids"],
                    alias_cache,
                ),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "latency_seconds": 0.0,
            }
            if selector_valid and not args.selection_only:
                context = selected_context(selected_ids, bank_by_id, representation)
                answer_started = time.perf_counter()
                raw_answer, answer_usage = e0.call_ollama(
                    args.base_url,
                    args.model,
                    answer_prompt(bank["question"], context, representation),
                    args.timeout,
                    args.retries,
                    args.answer_max_tokens,
                )
                answer_latency = time.perf_counter() - answer_started
                predicted, valid, parse_error = e0.parse_answers(raw_answer)
                row.update(
                    {
                        "answer_attempted": True,
                        "prediction": predicted,
                        "raw_response": raw_answer,
                        "output_valid": valid,
                        "parse_error": parse_error,
                        **e0.score_canonical_answer_set(predicted, bank["gold_answers"]),
                        **e0.score_alias_answer_set(
                            predicted,
                            bank["gold_answers"],
                            bank["gold_answer_ids"],
                            alias_cache,
                        ),
                        "usage": answer_usage,
                        "latency_seconds": answer_latency,
                    }
                )
            append_jsonl(checkpoint_path, row)
            results.append(row)
            print(
                f"Completed {completed}/{total} "
                f"({bank['hop_count']}-hop, {representation})",
                flush=True,
            )

    results.sort(
        key=lambda row: (row["hop_count"], row["dataset_index"], REPRESENTATIONS.index(row["representation"]))
    )
    write_jsonl(args.output_dir / "selection_results.jsonl", results)
    if not args.selection_only:
        write_jsonl(args.output_dir / "predictions.jsonl", results)
    report = {
        "status": "completed",
        "metrics_schema_version": 2,
        "condition": "E3_grailqa_manual_llm_retrieval",
        "model": args.model,
        "questions": len(banks),
        "cases": len(results),
        "selector_calls": len(results),
        "answer_calls": 0 if args.selection_only else sum(
            bool(row["answer_attempted"]) for row in results
        ),
        "selection_only": args.selection_only,
        "evaluation": e0.evaluation_metadata(alias_metadata),
        "configuration": {
            "noise_statements_per_question": args.noise_statements,
            "top_k": args.top_k,
            "gold_forced_into_selector_output": False,
            "temperature": 0,
            "think": False,
            "candidate_bundle_sha256": candidate_hash,
            "alias_cache": alias_metadata,
            "alias_cache_loaded_into_prompt": False,
        },
        "by_representation": {
            representation: summarize(
                [row for row in results if row["representation"] == representation],
                not args.selection_only,
            )
            for representation in REPRESENTATIONS
        },
        "by_hop": {
            str(hop): {
                "by_representation": {
                    representation: summarize(
                        [
                            row
                            for row in results
                            if row["hop_count"] == hop
                            and row["representation"] == representation
                        ],
                        not args.selection_only,
                    )
                    for representation in REPRESENTATIONS
                }
            }
            for hop in sorted({row["hop_count"] for row in results})
        },
    }
    write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
