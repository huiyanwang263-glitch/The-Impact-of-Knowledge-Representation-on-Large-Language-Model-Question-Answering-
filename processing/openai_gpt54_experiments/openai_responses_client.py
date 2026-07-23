#!/usr/bin/env python3
"""Minimal OpenAI Responses API adapter for the existing QA experiment scripts."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4-2026-03-05"
DEFAULT_REASONING_EFFORT = "none"


def _responses_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/responses"):
        return root
    return root + "/responses"


def _output_text(body: dict[str, Any]) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    texts: list[str] = []
    refusals: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                texts.append(content["text"])
            elif content.get("type") == "refusal" and isinstance(content.get("refusal"), str):
                refusals.append(content["refusal"])
    if texts:
        return "\n".join(texts).strip()
    if refusals:
        raise RuntimeError("OpenAI model refused the request: " + " ".join(refusals))
    raise RuntimeError("OpenAI response contained no output_text")


def _usage(body: dict[str, Any]) -> dict[str, Any]:
    usage = body.get("usage") or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    result = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": int(usage.get("total_tokens", input_tokens + output_tokens) or 0),
    }
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    result["cached_input_tokens"] = int(input_details.get("cached_tokens", 0) or 0)
    result["reasoning_tokens"] = int(output_details.get("reasoning_tokens", 0) or 0)
    result["response_id"] = str(body.get("id") or "")
    result["response_status"] = str(body.get("status") or "")
    return result


def call_responses(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    retries: int,
    max_output_tokens: int,
    *,
    schema: dict[str, Any] | None = None,
    schema_name: str = "qa_response",
) -> tuple[str, dict[str, Any]]:
    """Call Responses API and return text plus legacy-compatible token usage."""
    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not resolved_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Configure it in the shell; never paste it into source code."
        )

    payload: dict[str, Any] = {
        "model": model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        "input": messages,
        "max_output_tokens": max_output_tokens,
        "reasoning": {
            "effort": os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
        },
        "store": False,
    }
    if schema is not None:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        }

    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_url = _responses_url(base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_API_BASE))
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                request_url,
                data=encoded,
                headers={
                    "Authorization": f"Bearer {resolved_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            if body.get("status") == "incomplete":
                raise RuntimeError(
                    f"OpenAI response incomplete: {body.get('incomplete_details')}"
                )
            return _output_text(body), _usage(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenAI HTTP {exc.code}: {detail}")
            retryable = exc.code in {408, 409, 429} or exc.code >= 500
            if not retryable or attempt == retries:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt == retries:
                break
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenAI Responses request failed after {retries + 1} attempts: {last_error}")


def selector_schema(top_k: int) -> dict[str, Any]:
    return {
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
        "additionalProperties": False,
    }


def answer_set_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 128,
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    }


def call_kqapro_legacy(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    retries: int,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Compatibility function for KQA Pro's former Chat Completions helper."""
    combined = "\n".join(message.get("content", "") for message in messages)
    if "selected_statement_ids" in combined and "evidence selector" in combined.casefold():
        import re

        match = re.search(r"up to\s+(\d+)\s+statements", combined, re.IGNORECASE)
        top_k = int(match.group(1)) if match else 50
        return call_responses(
            base_url,
            api_key,
            model,
            messages,
            timeout,
            retries,
            max_tokens,
            schema=selector_schema(top_k),
            schema_name="evidence_selection",
        )
    return call_responses(
        base_url, api_key, model, messages, timeout, retries, max_tokens
    )


def call_grailqa_answer(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    retries: int,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    return call_responses(
        base_url,
        "",
        model,
        messages,
        timeout,
        retries,
        max_tokens,
        schema=answer_set_schema(),
        schema_name="answer_set",
    )


def call_grailqa_selector(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    top_k: int,
    timeout: int,
    retries: int,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    return call_responses(
        base_url,
        "",
        model,
        messages,
        timeout,
        retries,
        max_tokens,
        schema=selector_schema(top_k),
        schema_name="evidence_selection",
    )
