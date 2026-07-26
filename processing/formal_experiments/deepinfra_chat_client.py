#!/usr/bin/env python3
"""DeepInfra Chat Completions adapter for the formal QA experiments."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_API_BASE = "https://api.deepinfra.com/v1/openai"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-14B"
DEFAULT_MODEL_VERSION = "01b0452aeedc42eaf4a73d87654a8129d6a436f0"
DEFAULT_MODEL = f"{DEFAULT_MODEL_NAME}:{DEFAULT_MODEL_VERSION}"


def _chat_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return root + "/chat/completions"


def _usage(body: dict[str, Any]) -> dict[str, Any]:
    usage = body.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    reasoning_tokens = int(
        completion_details.get(
            "reasoning_tokens", usage.get("reasoning_tokens", 0)
        )
        or 0
    )
    result: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(
            usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
        ),
        "reasoning_tokens": reasoning_tokens,
        "response_id": str(body.get("id") or ""),
    }
    if usage.get("estimated_cost") is not None:
        result["estimated_cost"] = float(usage["estimated_cost"])
    return result


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


def call_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    retries: int,
    max_tokens: int,
    *,
    schema: dict[str, Any] | None = None,
    schema_name: str = "qa_response",
) -> tuple[str, dict[str, Any]]:
    """Call DeepInfra with deterministic decoding and thinking disabled."""
    api_key = os.getenv("DEEPINFRA_TOKEN", "")
    if not api_key:
        raise RuntimeError(
            "DEEPINFRA_TOKEN is not set. In PowerShell run: "
            '$env:DEEPINFRA_TOKEN = "your-token"'
        )

    payload: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": False,
        "reasoning_effort": "none",
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }

    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                _chat_url(base_url or DEFAULT_API_BASE),
                data=encoded,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            choices = body.get("choices") or []
            if not choices:
                raise RuntimeError("DeepInfra response has no choices")
            choice = choices[0]
            message = choice.get("message") or {}
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("DeepInfra response has no message content")
            usage = _usage(body)
            reasoning_content = message.get("reasoning_content")
            usage["reasoning_content_present"] = bool(reasoning_content)
            usage["thinking_disabled"] = (
                usage["reasoning_tokens"] == 0
                and not usage["reasoning_content_present"]
            )
            if not usage["thinking_disabled"]:
                raise RuntimeError(
                    "DeepInfra returned reasoning tokens/content with "
                    "reasoning_effort='none'"
                )
            usage["finish_reason"] = choice.get("finish_reason")
            usage["served_model"] = str(body.get("model") or "")
            if body.get("service_tier") is not None:
                usage["service_tier"] = body["service_tier"]
            return content.strip(), usage
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"DeepInfra HTTP {exc.code}: {detail}")
            retryable = exc.code in {408, 409, 429} or exc.code >= 500
            if not retryable or attempt == retries:
                break
        except (
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            last_error = exc
            if attempt == retries:
                break
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"DeepInfra request failed after {retries + 1} attempts: {last_error}"
    )


def call_kqapro_legacy(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    retries: int,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Compatibility function for the KQA Pro stage scripts."""
    del api_key
    combined = "\n".join(message.get("content", "") for message in messages)
    if "selected_statement_ids" in combined and "evidence selector" in combined.casefold():
        match = re.search(r"up to\s+(\d+)\s+statements", combined, re.IGNORECASE)
        top_k = int(match.group(1)) if match else 50
        return call_chat(
            base_url,
            model,
            messages,
            timeout,
            retries,
            max_tokens,
            schema=selector_schema(top_k),
            schema_name="evidence_selection",
        )
    return call_chat(base_url, model, messages, timeout, retries, max_tokens)


def call_grailqa_answer(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    retries: int,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    return call_chat(
        base_url,
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
    return call_chat(
        base_url,
        model,
        messages,
        timeout,
        retries,
        max_tokens,
        schema=selector_schema(top_k),
        schema_name="evidence_selection",
    )
