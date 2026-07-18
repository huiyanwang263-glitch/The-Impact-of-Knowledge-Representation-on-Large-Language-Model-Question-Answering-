#!/usr/bin/env python3
"""Materialize grounded 2/3-hop GrailQA gold paths through SPARQL.

The public GrailQA train/dev records contain gold query graphs and logical
forms, but their class nodes are variables rather than grounded Freebase
facts. This script rewrites each gold SPARQL query so that every graph node is
returned, executes it, validates the question-node denotation against the
annotated answer set, and emits a canonical statement layer suitable for
aligned Document/Triple/Graph experiments.

Only non-aggregate questions (``function == none``) are accepted by design.
The implementation uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DEFAULT_DATA = (
    PROJECT
    / "GrailQA_v1.0 (1)"
    / "GrailQA_v1.0"
    / "grailqa_v1.0_dev.json"
)
DEFAULT_ENDPOINT = os.environ.get(
    "GRAILQA_SPARQL_ENDPOINT",
    "http://cse-cnc238814s.cse.ohio-state.edu:3093/sparql",
)
FREEBASE_NS = "http://rdf.freebase.com/ns/"
SELECT_DISTINCT_RE = re.compile(
    r"\bSELECT\s+DISTINCT\s+.*?\bWHERE\s*\{", re.IGNORECASE | re.DOTALL
)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, value: Any, length: int = 16) -> str:
    digest = hashlib.sha1(compact_json(value).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


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


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def matching_brace(text: str, opening: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    quote = ""
    for index in range(opening, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Unbalanced braces in SPARQL query")


def materialization_query(item: dict[str, Any]) -> tuple[str, list[str]]:
    """Replace GrailQA's answer-only outer SELECT with an all-node SELECT."""
    source = item["sparql_query"]
    matches = list(SELECT_DISTINCT_RE.finditer(source))
    if not matches:
        raise ValueError("No SELECT DISTINCT ... WHERE block found")

    # GrailQA normally wraps the useful DISTINCT query in an answer-only SELECT.
    # The final SELECT DISTINCT block is the innermost graph-pattern query.
    inner = matches[-1]
    opening = source.find("{", inner.start(), inner.end())
    closing = matching_brace(source, opening)
    body = source[opening + 1 : closing].strip()
    first_select = re.search(r"\bSELECT\b", source, re.IGNORECASE)
    if first_select is None:
        raise ValueError("No SELECT clause found")
    prefix = source[: first_select.start()].strip()

    node_variables = [f"x{int(node['nid'])}" for node in item["graph_query"]["nodes"]]
    node_variables = sorted(set(node_variables), key=lambda name: int(name[1:]))
    selected = " ".join(f"?{name}" for name in node_variables)
    query = f"{prefix}\nSELECT DISTINCT {selected} WHERE {{\n{body}\n}}"
    return query.strip(), node_variables


def endpoint_query(endpoint: str, query: str, timeout: float) -> dict[str, Any]:
    payload = urllib.parse.urlencode(
        {"query": query, "format": "application/sparql-results+json"}
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "User-Agent": "grailqa-gold-path-builder/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def query_with_retries(
    endpoint: str, query: str, timeout: float, retries: int, retry_backoff: float
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return endpoint_query(endpoint, query, timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_backoff * (attempt + 1))
    assert last_error is not None
    raise last_error


def normalized_binding(binding: dict[str, Any]) -> dict[str, Any]:
    raw = str(binding.get("value", ""))
    binding_type = binding.get("type", "literal")
    if binding_type == "uri":
        node_id = raw[len(FREEBASE_NS) :] if raw.startswith(FREEBASE_NS) else raw
        return {"id": node_id, "label": node_id, "kind": "entity", "value": raw}

    datatype = binding.get("datatype")
    language = binding.get("xml:lang")
    identity = raw
    if datatype:
        identity = f"{raw}^^{datatype}"
    elif language:
        identity = f"{raw}@{language}"
    result = {"id": identity, "label": raw, "kind": "literal", "value": raw}
    if datatype:
        result["datatype"] = datatype
    if language:
        result["language"] = language
    return result


def normalize_answer_id(value: str) -> str:
    normalized = value
    if normalized.startswith(FREEBASE_NS):
        normalized = normalized[len(FREEBASE_NS) :]
    return normalized.replace("-08:00", "")


def parse_bindings(
    response: dict[str, Any], variables: list[str]
) -> tuple[list[dict[str, dict[str, Any]]], int]:
    complete: list[dict[str, dict[str, Any]]] = []
    incomplete = 0
    for raw_row in response.get("results", {}).get("bindings", []):
        if any(variable not in raw_row for variable in variables):
            incomplete += 1
            continue
        complete.append(
            {variable: normalized_binding(raw_row[variable]) for variable in variables}
        )

    deduplicated: dict[str, dict[str, dict[str, Any]]] = {}
    for row in complete:
        key = compact_json({name: value["id"] for name, value in row.items()})
        deduplicated[key] = row
    return list(deduplicated.values()), incomplete


def question_variable(item: dict[str, Any]) -> str:
    nodes = [node for node in item["graph_query"]["nodes"] if node.get("question_node") == 1]
    if len(nodes) != 1:
        raise ValueError(f"Expected one question node, found {len(nodes)}")
    return f"x{int(nodes[0]['nid'])}"


def validate_denotation(
    item: dict[str, Any], bindings: list[dict[str, dict[str, Any]]]
) -> tuple[bool, set[str], set[str]]:
    variable = question_variable(item)
    observed = {normalize_answer_id(row[variable]["id"]) for row in bindings}
    expected = {
        normalize_answer_id(answer["answer_argument"]) for answer in item.get("answer", [])
    }
    return observed == expected, expected, observed


def initial_labels(item: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for node in item["graph_query"]["nodes"]:
        if node.get("node_type") in {"entity", "literal"}:
            labels[node["id"]] = node.get("friendly_name") or node["id"]
    for answer in item.get("answer", []):
        labels[normalize_answer_id(answer["answer_argument"])] = (
            answer.get("entity_name") or answer["answer_argument"]
        )
    return labels


def freebase_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", value))


def fetch_labels(
    endpoint: str,
    entity_ids: set[str],
    timeout: float,
    retries: int,
    retry_backoff: float,
    batch_size: int,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    labels: dict[str, str] = {}
    failures: list[dict[str, Any]] = []
    ids = sorted(entity_id for entity_id in entity_ids if freebase_identifier(entity_id))
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        values = " ".join(f":{entity_id}" for entity_id in batch)
        query = f"""PREFIX : <{FREEBASE_NS}>
SELECT ?entity ?name WHERE {{
  VALUES ?entity {{ {values} }}
  ?entity :type.object.name ?name .
  FILTER(lang(?name) = \"\" || langMatches(lang(?name), \"en\"))
}}"""
        try:
            response = query_with_retries(
                endpoint, query, timeout, retries, retry_backoff
            )
        except Exception as exc:  # label lookup is non-fatal
            failures.append({"entity_ids": batch, "error": repr(exc)})
            continue
        for row in response.get("results", {}).get("bindings", []):
            if "entity" not in row or "name" not in row:
                continue
            entity = normalize_answer_id(row["entity"]["value"])
            name = row["name"]["value"]
            labels.setdefault(entity, name)
    return labels, failures


def node_ref(node: dict[str, Any], labels: dict[str, str]) -> dict[str, Any]:
    result = dict(node)
    result["label"] = labels.get(node["id"], node.get("label") or node["id"])
    result.pop("value", None)
    return result


def build_statement(
    edge: dict[str, Any],
    binding: dict[str, dict[str, Any]],
    labels: dict[str, str],
) -> dict[str, Any]:
    subject = node_ref(binding[f"x{int(edge['start'])}"], labels)
    obj = node_ref(binding[f"x{int(edge['end'])}"], labels)
    identity = {
        "subject_id": subject["id"],
        "predicate": edge["relation"],
        "object_id": obj["id"],
    }
    return {
        "statement_id": stable_id("gstmt", identity),
        "kind": "relation",
        "subject": subject,
        "predicate": edge["relation"],
        "predicate_label": edge.get("friendly_name") or edge["relation"],
        "object": obj,
        "qualifiers": [],
    }


def build_question_record(
    item: dict[str, Any],
    dataset_index: int,
    query: str,
    bindings: list[dict[str, dict[str, Any]]],
    labels: dict[str, str],
    incomplete_bindings: int,
) -> dict[str, Any]:
    statements: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        for edge in item["graph_query"]["edges"]:
            statement = build_statement(edge, binding, labels)
            statements[statement["statement_id"]] = statement
    ordered = sorted(statements.values(), key=lambda row: row["statement_id"])
    answer_ids = [normalize_answer_id(answer["answer_argument"]) for answer in item["answer"]]
    return {
        "question_id": str(item["qid"]),
        "dataset_index": dataset_index,
        "hop_count": int(item["num_edge"]),
        "question": item["question"],
        "answers": item["answer"],
        "answer_ids": answer_ids,
        "level": item.get("level"),
        "domains": item.get("domains", []),
        "function": item.get("function"),
        "graph_query": item["graph_query"],
        "s_expression": item.get("s_expression"),
        "source_sparql": item["sparql_query"],
        "materialization_query": query,
        "binding_count": len(bindings),
        "incomplete_binding_count": incomplete_bindings,
        "gold_statement_ids": [statement["statement_id"] for statement in ordered],
        "gold_statements": ordered,
    }


def select_candidates(
    data: list[dict[str, Any]], hops: set[int], max_per_hop: int | None
) -> tuple[list[tuple[int, dict[str, Any]]], Counter[str]]:
    counts: Counter[str] = Counter()
    selected: list[tuple[int, dict[str, Any]]] = []
    per_hop: Counter[int] = Counter()
    for index, item in enumerate(data):
        hop = int(item.get("num_edge", -1))
        if hop not in hops:
            counts["excluded_hop"] += 1
            continue
        if item.get("function") != "none":
            counts["excluded_function"] += 1
            continue
        if not item.get("answer"):
            counts["excluded_empty_answer"] += 1
            continue
        if any(answer.get("answer_type") != "Entity" for answer in item["answer"]):
            counts["excluded_non_entity_answer"] += 1
            continue
        if max_per_hop is not None and per_hop[hop] >= max_per_hop:
            counts["excluded_max_per_hop"] += 1
            continue
        selected.append((index, item))
        per_hop[hop] += 1
        counts[f"selected_{hop}_hop"] += 1
    counts["selected_total"] = len(selected)
    return selected, counts


def checkpoint_successes(path: Path) -> dict[str, dict[str, Any]]:
    successes: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") == "success":
            successes[str(row["question_id"])] = row
    return successes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "dev_hops_2_3")
    parser.add_argument("--hops", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--max-per-hop", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--max-consecutive-endpoint-failures", type=int, default=3)
    parser.add_argument("--label-batch-size", type=int, default=100)
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    if any(hop not in {2, 3} for hop in args.hops):
        raise ValueError("This controlled builder accepts only 2-hop and 3-hop questions")

    data = json.loads(args.data.read_text(encoding="utf-8"))
    candidates, selection_counts = select_candidates(
        data, set(args.hops), args.max_per_hop
    )
    prepared: list[tuple[int, dict[str, Any], str, list[str]]] = []
    preparation_failures: list[dict[str, Any]] = []
    for dataset_index, item in candidates:
        try:
            query, variables = materialization_query(item)
            prepared.append((dataset_index, item, query, variables))
        except Exception as exc:
            preparation_failures.append(
                {
                    "question_id": str(item.get("qid")),
                    "dataset_index": dataset_index,
                    "stage": "query_rewrite",
                    "error": repr(exc),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.output_dir / "materialization_queries.jsonl",
        (
            {
                "question_id": str(item["qid"]),
                "dataset_index": index,
                "hop_count": item["num_edge"],
                "question": item["question"],
                "variables": variables,
                "query": query,
            }
            for index, item, query, variables in prepared
        ),
    )
    selection_summary = {
        "data": str(args.data.resolve()),
        "requested_hops": sorted(set(args.hops)),
        "selection": dict(selection_counts),
        "queries_prepared": len(prepared),
        "query_rewrite_failures": len(preparation_failures),
        "dry_run": args.dry_run,
    }
    write_json(args.output_dir / "selection_summary.json", selection_summary)
    if args.dry_run:
        write_jsonl(args.output_dir / "failures.jsonl", preparation_failures)
        print(json.dumps(selection_summary, ensure_ascii=False, indent=2))
        return

    checkpoint = args.output_dir / "query_results.jsonl"
    if args.no_resume and checkpoint.exists():
        raise RuntimeError(
            f"{checkpoint} exists; choose a new output directory instead of overwriting it"
        )
    cached = {} if args.no_resume else checkpoint_successes(checkpoint)
    latest_failures: dict[str, dict[str, Any]] = {
        row["question_id"]: row for row in preparation_failures
    }
    successful: dict[str, dict[str, Any]] = dict(cached)
    consecutive_endpoint_failures = 0
    endpoint_abort_reason: str | None = None

    for position, (dataset_index, item, query, variables) in enumerate(prepared, start=1):
        question_id = str(item["qid"])
        query_hash = stable_id("query", query)
        cached_row = successful.get(question_id)
        if cached_row and cached_row.get("query_hash") == query_hash:
            continue
        try:
            response = query_with_retries(
                args.endpoint, query, args.timeout, args.retries, args.retry_backoff
            )
            consecutive_endpoint_failures = 0
            bindings, incomplete = parse_bindings(response, variables)
            valid, expected, observed = validate_denotation(item, bindings)
            if not bindings:
                raise ValueError("SPARQL returned no complete node bindings")
            if not valid:
                raise ValueError(
                    "Denotation mismatch: "
                    f"expected={sorted(expected)!r}, observed={sorted(observed)!r}"
                )
            row = {
                "status": "success",
                "question_id": question_id,
                "dataset_index": dataset_index,
                "query_hash": query_hash,
                "variables": variables,
                "bindings": bindings,
                "incomplete_bindings": incomplete,
            }
            append_jsonl(checkpoint, row)
            successful[question_id] = row
            latest_failures.pop(question_id, None)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            consecutive_endpoint_failures += 1
            row = {
                "status": "failure",
                "question_id": question_id,
                "dataset_index": dataset_index,
                "query_hash": query_hash,
                "stage": "endpoint",
                "error": repr(exc),
            }
            append_jsonl(checkpoint, row)
            latest_failures[question_id] = row
            if consecutive_endpoint_failures >= args.max_consecutive_endpoint_failures:
                endpoint_abort_reason = (
                    f"Aborted after {consecutive_endpoint_failures} consecutive "
                    f"endpoint failures; last error: {exc!r}"
                )
                print(endpoint_abort_reason, file=sys.stderr, flush=True)
                break
        except Exception as exc:
            consecutive_endpoint_failures = 0
            row = {
                "status": "failure",
                "question_id": question_id,
                "dataset_index": dataset_index,
                "query_hash": query_hash,
                "stage": "sparql_or_validation",
                "error": repr(exc),
            }
            append_jsonl(checkpoint, row)
            latest_failures[question_id] = row
        if position % 25 == 0 or position == len(prepared):
            print(
                f"Processed {position}/{len(prepared)}; "
                f"success={len(successful)}, current_failures={len(latest_failures)}",
                flush=True,
            )

    by_id = {str(item["qid"]): (index, item, query) for index, item, query, _ in prepared}
    labels: dict[str, str] = {}
    entity_ids: set[str] = set()
    for question_id, result in successful.items():
        if question_id not in by_id:
            continue
        _, item, _ = by_id[question_id]
        labels.update(initial_labels(item))
        for binding in result["bindings"]:
            for node in binding.values():
                if node["kind"] == "entity":
                    entity_ids.add(node["id"])

    label_failures: list[dict[str, Any]] = []
    if not args.skip_labels:
        fetched, label_failures = fetch_labels(
            args.endpoint,
            entity_ids - set(labels),
            args.timeout,
            args.retries,
            args.retry_backoff,
            args.label_batch_size,
        )
        labels.update(fetched)

    questions: list[dict[str, Any]] = []
    canonical: dict[str, dict[str, Any]] = {}
    for question_id, result in successful.items():
        if question_id not in by_id:
            continue
        dataset_index, item, query = by_id[question_id]
        record = build_question_record(
            item,
            dataset_index,
            query,
            result["bindings"],
            labels,
            result.get("incomplete_bindings", 0),
        )
        questions.append(record)
        for statement in record["gold_statements"]:
            canonical[statement["statement_id"]] = statement

    questions.sort(key=lambda row: (row["hop_count"], row["dataset_index"]))
    statements = sorted(canonical.values(), key=lambda row: row["statement_id"])
    write_jsonl(args.output_dir / "gold_questions.jsonl", questions)
    write_jsonl(args.output_dir / "canonical_gold_statements.jsonl", statements)
    write_jsonl(
        args.output_dir / "failures.jsonl",
        sorted(latest_failures.values(), key=lambda row: row.get("dataset_index", -1)),
    )
    write_jsonl(args.output_dir / "label_failures.jsonl", label_failures)

    hop_counts = Counter(str(row["hop_count"]) for row in questions)
    level_counts = Counter(row.get("level") or "missing" for row in questions)
    unlabeled = sorted(entity_id for entity_id in entity_ids if entity_id not in labels)
    summary = {
        **selection_summary,
        "dry_run": False,
        "endpoint": args.endpoint,
        "materialized_questions": len(questions),
        "materialized_by_hop": dict(hop_counts),
        "materialized_by_level": dict(level_counts),
        "canonical_statement_count": len(statements),
        "current_failure_count": len(latest_failures),
        "endpoint_abort_reason": endpoint_abort_reason,
        "label_lookup_failure_batches": len(label_failures),
        "entities_without_friendly_label": len(unlabeled),
        "entities_without_friendly_label_sample": unlabeled[:20],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed query checkpoints are preserved.", file=sys.stderr)
        raise SystemExit(130)
