#!/usr/bin/env python3
"""Shared loader and output metadata helpers for GPT-5.4 experiment wrappers."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from openai_responses_client import DEFAULT_API_BASE, DEFAULT_MODEL


HERE = Path(__file__).resolve().parent
PROCESSING = HERE.parent


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load experiment script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def has_flag(arguments: list[str], flag: str) -> bool:
    return flag in arguments or any(value.startswith(flag + "=") for value in arguments)


def option_value(arguments: list[str], flag: str) -> str | None:
    for index, value in enumerate(arguments):
        if value == flag and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return None


def require_key(build_only: bool) -> None:
    if not build_only and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. In PowerShell run: "
            "$env:OPENAI_API_KEY = \"sk-...\""
        )


def invoke(module: ModuleType, arguments: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(module.__file__).resolve()), *arguments]
        module.main()
    finally:
        sys.argv = old_argv


def annotate_output(
    output_dir: Path,
    model: str,
    api_base: str,
    reasoning_effort: str,
) -> None:
    provider = {
        "provider": "openai",
        "api": "responses",
        "model": model,
        "api_base": api_base,
        "reasoning_effort": reasoning_effort,
        "temperature": None,
        "store": False,
        "structured_outputs_for_json_stages": True,
    }
    for filename in ("configuration.json", "metrics.json"):
        path = output_dir / filename
        if not path.exists():
            continue
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        payload["openai_provider"] = provider
        configuration = payload.get("configuration")
        if filename == "configuration.json" and not isinstance(configuration, dict):
            configuration = payload
        if isinstance(configuration, dict):
            for key in ("temperature", "selector_temperature", "answer_temperature"):
                if key in configuration:
                    configuration[key] = None
            configuration["reasoning_effort"] = reasoning_effort
            configuration["api"] = "responses"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def common_defaults() -> tuple[str, str, str]:
    return (
        os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        os.getenv("OPENAI_BASE_URL", DEFAULT_API_BASE),
        os.getenv("OPENAI_REASONING_EFFORT", "none"),
    )
