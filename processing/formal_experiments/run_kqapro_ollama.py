#!/usr/bin/env python3
"""Run a frozen KQA Pro manifest through the existing Ollama E0-E3 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
GPT_DIR = HERE.parent / "openai_gpt54_experiments"
sys.path.insert(0, str(GPT_DIR))
import run_kqapro_openai as shared  # noqa: E402
from runner_common import has_flag, invoke, load_module, option_value  # noqa: E402
sys.path.insert(0, str(HERE))
import kqapro_formal_noise as formal_noise  # noqa: E402


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Exact final answer entity or value only; no reasoning or explanation.",
            "maxLength": 160,
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def native_no_think_answer(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    retries: int,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Call Ollama natively with thinking disabled and a concise answer schema."""
    del api_key
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = endpoint[:-3]
    endpoint += "/api/chat"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": ANSWER_SCHEMA,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = str(body["message"]["content"]).strip()
            parsed = json.loads(content)
            if not isinstance(parsed, dict) or not isinstance(
                parsed.get("answer"), str
            ):
                raise ValueError("Ollama structured answer is missing string field 'answer'")
            prompt_tokens = int(body.get("prompt_eval_count", 0))
            completion_tokens = int(body.get("eval_count", 0))
            return content, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "thinking_tokens": 0,
                "thinking_disabled": True,
                "done_reason": body.get("done_reason"),
            }
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(
        f"Ollama native no-thinking request failed after {retries + 1} attempts: "
        f"{last_error}"
    )


def patch_answer_client(stage: str, module: object) -> None:
    if stage == "e0":
        module.oracle.call_chat_completions = native_no_think_answer
    elif stage == "e1":
        module.base.call_chat_completions = native_no_think_answer
    elif stage == "e3":
        module.oracle_base.call_chat_completions = native_no_think_answer
    elif stage == "e2":
        module.e3.oracle_base.call_chat_completions = native_no_think_answer


def annotate_no_think_output(output_dir: Path) -> None:
    provider = {
        "provider": "ollama",
        "api": "native_chat",
        "thinking": False,
        "structured_answer_schema": ANSWER_SCHEMA,
    }
    for filename in ("configuration.json", "metrics.json"):
        path = output_dir / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["ollama_provider"] = provider
        configuration = payload.get("configuration")
        if filename == "configuration.json" and not isinstance(configuration, dict):
            configuration = payload
        if isinstance(configuration, dict):
            configuration["answer_thinking"] = False
            configuration["answer_response_format"] = "json_schema"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Unknown arguments are forwarded to the existing stage script.",
    )
    parser.add_argument("stage", choices=shared.TARGETS)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-4b-16k")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "output" / "runs" / "kqapro_ollama",
    )
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, forwarded = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    module = load_module(shared.TARGETS[args.stage], f"formal_kqapro_ollama_{args.stage}")
    patch_answer_client(args.stage, module)
    if args.stage == "e3":
        module.build_question_derived_noise_bank = formal_noise.make_adaptive_builder(
            module
        )

    stage_output = args.output_root / args.stage
    original_output = option_value(forwarded, "--output-dir")
    if original_output is not None:
        stage_output = Path(original_output)
    else:
        forwarded += ["--output-dir", str(stage_output)]
    forwarded += ["--base-url", args.base_url, "--model", args.model]
    if args.build_only:
        forwarded.append("--build-only")

    token_defaults = {
        "e0": ("--max-tokens", "256"),
        "e1": ("--max-tokens", "256"),
        "e2": ("--answer-max-tokens", "256"),
    }
    if args.stage in token_defaults:
        flag, value = token_defaults[args.stage]
        if not has_flag(forwarded, flag):
            forwarded += [flag, value]
    if args.stage == "e3":
        if not has_flag(forwarded, "--selector-api"):
            forwarded += ["--selector-api", "ollama-native"]
        if not has_flag(forwarded, "--selector-max-tokens"):
            forwarded += ["--selector-max-tokens", "512"]
        if not has_flag(forwarded, "--answer-max-tokens"):
            forwarded += ["--answer-max-tokens", "256"]

    if args.stage in {"e0", "e2", "e3"} and not has_flag(forwarded, "--questions"):
        forwarded += ["--questions", str(args.manifest)]
    if args.stage == "e2":
        dependencies = {
            "--candidate-banks": args.output_root / "e3" / "candidate_banks.jsonl",
            "--selection-results": args.output_root / "e3" / "selection_results.jsonl",
            "--e3-configuration": args.output_root / "e3" / "configuration.json",
        }
        for flag, path in dependencies.items():
            if not has_flag(forwarded, flag):
                forwarded += [flag, str(path)]

    counts: dict[int, int] | None = None
    if args.stage == "e3":
        counts = shared.configure_e3_manifest(module, args.manifest)
    if args.stage == "e1":
        shared.run_e1_manifest(module, args.manifest, forwarded)
    else:
        invoke(module, forwarded)
    if counts is not None:
        shared.correct_e3_metadata(stage_output, counts)
        formal_noise.annotate_output(stage_output)
    annotate_no_think_output(stage_output)


if __name__ == "__main__":
    main()
