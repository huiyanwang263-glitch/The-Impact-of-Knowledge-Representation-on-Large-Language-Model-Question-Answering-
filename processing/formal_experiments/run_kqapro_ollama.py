#!/usr/bin/env python3
"""Run a frozen KQA Pro manifest through the existing Ollama E0-E3 pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
GPT_DIR = HERE.parent / "openai_gpt54_experiments"
sys.path.insert(0, str(GPT_DIR))
import run_kqapro_openai as shared  # noqa: E402
from runner_common import has_flag, invoke, load_module, option_value  # noqa: E402
sys.path.insert(0, str(HERE))
import kqapro_formal_noise as formal_noise  # noqa: E402


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
        "e0": ("--max-tokens", "4096"),
        "e1": ("--max-tokens", "4096"),
        "e2": ("--answer-max-tokens", "4096"),
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
            forwarded += ["--answer-max-tokens", "4096"]

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


if __name__ == "__main__":
    main()
