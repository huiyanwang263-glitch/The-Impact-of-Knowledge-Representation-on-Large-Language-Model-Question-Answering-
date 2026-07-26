#!/usr/bin/env python3
"""Audit and annotate one completed DeepInfra experiment stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


def _usage_records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if (
            isinstance(value.get("response_id"), str)
            and value["response_id"]
            and "prompt_tokens" in value
            and "completion_tokens" in value
        ):
            yield value
        for nested in value.values():
            yield from _usage_records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _usage_records(nested)


def aggregate_usage(output_dir: Path) -> dict[str, Any]:
    """Deduplicate usage copied across checkpoints and final JSONL outputs."""
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for usage in _usage_records(row):
                    records.setdefault(str(usage["response_id"]), usage)

    prompt_tokens = sum(int(row.get("prompt_tokens", 0) or 0) for row in records.values())
    completion_tokens = sum(
        int(row.get("completion_tokens", 0) or 0) for row in records.values()
    )
    reasoning_tokens = sum(
        int(row.get("reasoning_tokens", 0) or 0) for row in records.values()
    )
    estimated_costs = [
        float(row["estimated_cost"])
        for row in records.values()
        if row.get("estimated_cost") is not None
    ]
    response_id_digest = hashlib.sha256()
    for response_id in sorted(records):
        response_id_digest.update(response_id.encode("utf-8"))
        response_id_digest.update(b"\n")
    return {
        "unique_model_calls": len(records),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "estimated_cost_usd": sum(estimated_costs),
        "costed_call_count": len(estimated_costs),
        "served_models": sorted(
            {
                str(row["served_model"])
                for row in records.values()
                if row.get("served_model")
            }
        ),
        "service_tiers": sorted(
            {
                str(row["service_tier"])
                for row in records.values()
                if row.get("service_tier")
            }
        ),
        "finish_reasons": sorted(
            {
                str(row["finish_reason"])
                for row in records.values()
                if row.get("finish_reason")
            }
        ),
        "reasoning_content_present": any(
            bool(row.get("reasoning_content_present")) for row in records.values()
        ),
        "response_ids_sha256": response_id_digest.hexdigest(),
    }


def annotate_output(output_dir: Path, model: str, api_base: str) -> None:
    requested_name, separator, requested_version = model.partition(":")
    usage = aggregate_usage(output_dir)
    expected_served_model = requested_name
    served_model_consistent = usage["served_models"] in ([], [expected_served_model])
    thinking_disabled_verified = (
        usage["reasoning_tokens"] == 0
        and not usage["reasoning_content_present"]
    )
    provider = {
        "provider": "deepinfra",
        "api": "chat_completions",
        "requested_model": model,
        "requested_model_name": requested_name,
        "requested_model_version": requested_version if separator else None,
        "api_base": api_base,
        "reasoning_effort": "none",
        "thinking": False,
        "thinking_disabled_verified": thinking_disabled_verified,
        "served_model_consistent": served_model_consistent,
        "temperature": 0,
        "seed": 42,
        "structured_outputs_for_json_stages": True,
    }
    for filename in ("configuration.json", "metrics.json"):
        path = output_dir / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["deepinfra_provider"] = provider
        payload["deepinfra_usage"] = usage
        configuration = payload.get("configuration")
        if filename == "configuration.json" and not isinstance(configuration, dict):
            configuration = payload
        if isinstance(configuration, dict):
            configuration["reasoning_effort"] = "none"
            configuration["thinking"] = False
            configuration["api"] = "chat_completions"
            for key in ("temperature", "selector_temperature", "answer_temperature"):
                if key in configuration:
                    configuration[key] = 0
            configuration["seed"] = 42
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if not served_model_consistent:
        raise RuntimeError(
            f"DeepInfra served unexpected model(s): {usage['served_models']}; "
            f"expected {expected_served_model}"
        )
    if not thinking_disabled_verified:
        raise RuntimeError(
            "DeepInfra returned reasoning tokens/content although thinking was disabled"
        )
