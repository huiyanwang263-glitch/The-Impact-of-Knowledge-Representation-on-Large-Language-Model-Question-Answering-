#!/usr/bin/env python3
"""Run formal GrailQA E0-E3 with DeepInfra Qwen3-14B."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROCESSING = HERE.parent
GPT_DIR = PROCESSING / "openai_gpt54_experiments"
sys.path.insert(0, str(GPT_DIR))
from runner_common import has_flag, invoke, load_module, option_value  # noqa: E402

sys.path.insert(0, str(HERE))
from deepinfra_chat_client import (  # noqa: E402
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    call_grailqa_answer,
    call_grailqa_selector,
)
from deepinfra_run_metadata import annotate_output  # noqa: E402


TARGETS = {
    "e0": PROCESSING / "grailqa_e0_processing" / "run_e0_closed_book.py",
    "e1": PROCESSING / "grailqa_e1_gold_only_processing" / "run_e1_gold_only.py",
    "e2": PROCESSING / "grailqa_e2_forced_gold_processing" / "run_e2_forced_gold.py",
    "e3": (
        PROCESSING
        / "grailqa_e3_manual_llm_retrieval_processing"
        / "run_e3_manual_retrieval.py"
    ),
}


def require_key(build_only: bool) -> None:
    if not build_only and not os.getenv("DEEPINFRA_TOKEN"):
        raise RuntimeError(
            "DEEPINFRA_TOKEN is missing. In PowerShell run: "
            '$env:DEEPINFRA_TOKEN = "your-token"'
        )


def patch(stage: str, module: object) -> None:
    if stage == "e0":
        module.call_ollama = call_grailqa_answer
    elif stage == "e1":
        module.e0.call_ollama = call_grailqa_answer
    elif stage == "e3":
        module.call_selector = call_grailqa_selector
        module.e0.call_ollama = call_grailqa_answer
    elif stage == "e2":
        module.e3.call_selector = call_grailqa_selector
        module.e3.e0.call_ollama = call_grailqa_answer


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=TARGETS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "output" / "runs" / "grailqa_deepinfra",
    )
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, forwarded = parse_args()
    require_key(args.build_only)
    module = load_module(TARGETS[args.stage], f"deepinfra_grailqa_{args.stage}")
    patch(args.stage, module)

    stage_output = args.output_root / args.stage
    original_output = option_value(forwarded, "--output-dir")
    if original_output is not None:
        stage_output = Path(original_output)
    else:
        forwarded += ["--output-dir", str(stage_output)]
    forwarded += ["--base-url", args.api_base, "--model", args.model]
    if args.build_only:
        forwarded.append("--build-only")
    if args.stage in {"e0", "e1"} and not has_flag(forwarded, "--max-tokens"):
        forwarded += ["--max-tokens", "512"]
    if args.stage in {"e2", "e3"}:
        if not has_flag(forwarded, "--selector-max-tokens"):
            forwarded += ["--selector-max-tokens", "1024"]
        if not has_flag(forwarded, "--answer-max-tokens"):
            forwarded += ["--answer-max-tokens", "512"]
    if args.stage == "e2" and not has_flag(forwarded, "--e3-selection-results"):
        forwarded += [
            "--e3-selection-results",
            str(args.output_root / "e3" / "selection_results.jsonl"),
        ]

    invoke(module, forwarded)
    annotate_output(stage_output, args.model, args.api_base)


if __name__ == "__main__":
    main()
