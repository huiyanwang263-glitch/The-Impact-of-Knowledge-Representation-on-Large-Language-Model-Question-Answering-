#!/usr/bin/env python3
"""Run existing GrailQA E0/E1/E2/E3 logic with OpenAI GPT-5.4 Responses API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai_responses_client import call_grailqa_answer, call_grailqa_selector
from runner_common import (
    PROCESSING,
    annotate_output,
    common_defaults,
    has_flag,
    invoke,
    load_module,
    option_value,
    require_key,
)


TARGETS = {
    "e0": PROCESSING / "grailqa_e0_processing" / "run_e0_closed_book.py",
    "e1": PROCESSING / "grailqa_e1_gold_only_processing" / "run_e1_gold_only.py",
    "e2": PROCESSING / "grailqa_e2_forced_gold_processing" / "run_e2_forced_gold.py",
    "e3": PROCESSING / "grailqa_e3_manual_llm_retrieval_processing" / "run_e3_manual_retrieval.py",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    default_model, default_base, default_effort = common_defaults()
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Unknown arguments are forwarded to the original stage script.",
    )
    parser.add_argument("stage", choices=TARGETS)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--api-base", default=default_base)
    parser.add_argument("--reasoning-effort", default=default_effort, choices=["none"])
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "output" / "grailqa")
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_known_args()


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


def main() -> None:
    args, forwarded = parse_args()
    require_key(args.build_only)
    os.environ["OPENAI_REASONING_EFFORT"] = args.reasoning_effort

    module = load_module(TARGETS[args.stage], f"gpt54_grailqa_{args.stage}")
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
    annotate_output(stage_output, args.model, args.api_base, args.reasoning_effort)


if __name__ == "__main__":
    main()
