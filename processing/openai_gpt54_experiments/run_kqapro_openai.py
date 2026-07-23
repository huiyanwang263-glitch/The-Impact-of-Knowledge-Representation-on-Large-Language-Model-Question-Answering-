#!/usr/bin/env python3
"""Run existing KQA Pro E0/E1/E2/E3 logic with OpenAI GPT-5.4 Responses API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from openai_responses_client import call_kqapro_legacy
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
    "e0": PROCESSING / "closed_book_processing" / "run_e0_closed_book_hops_15.py",
    "e1": PROCESSING / "knowledge_representation_processing" / "run_oracle_hops_15.py",
    "e2": PROCESSING / "e2_forced_gold_processing" / "run_e2_forced_gold_hops_15.py",
    "e3": PROCESSING / "e3_manual_llm_retrieval_processing" / "run_e3_manual_llm_retrieval_hops_15.py",
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
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "output" / "kqapro")
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Frozen oracle_questions.jsonl shared by E0/E1/E2/E3. This is the "
            "recommended input for formal non-uniform hop counts."
        ),
    )
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_known_args()


def patch(stage: str, module: object) -> None:
    if stage == "e0":
        module.oracle.call_chat_completions = call_kqapro_legacy
    elif stage == "e1":
        module.base.call_chat_completions = call_kqapro_legacy
    elif stage == "e3":
        module.oracle_base.call_chat_completions = call_kqapro_legacy
    elif stage == "e2":
        module.e3.oracle_base.call_chat_completions = call_kqapro_legacy


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise RuntimeError(f"Empty KQA Pro manifest: {path}")
    required = {
        "question_id",
        "hop_count",
        "question",
        "choices",
        "answer",
        "gold_statement_ids",
        "gold_statements",
        "contexts",
    }
    for row in rows:
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"{row.get('question_id')}: manifest fields missing: {sorted(missing)}")
    return rows


def configure_e3_manifest(module: object, manifest: Path) -> dict[int, int]:
    rows = read_manifest(manifest)
    counts = dict(Counter(int(row["hop_count"]) for row in rows))
    module.EXPECTED_HOP_COUNTS = counts

    def validate(questions: list[dict]) -> None:
        actual = dict(Counter(int(item["hop_count"]) for item in questions))
        if actual != counts:
            raise RuntimeError(f"Expected hop counts {counts}, found {actual}")
        ids = [str(item["question_id"]) for item in questions]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Evaluation question IDs are not unique")
        for item in questions:
            gold_ids = item.get("gold_statement_ids", [])
            gold = item.get("gold_statements", [])
            if len(gold_ids) != int(item["hop_count"]) or len(gold) != len(gold_ids):
                raise RuntimeError(
                    f"{item['question_id']}: hop count and Gold statement count disagree"
                )
            if gold_ids != [statement["statement_id"] for statement in gold]:
                raise RuntimeError(f"{item['question_id']}: Gold IDs and statements disagree")

    module.validate_evaluation_questions = validate
    return counts


def run_e1_manifest(module: object, manifest: Path, arguments: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(module.__file__).resolve()), *arguments]
        stage_args = module.parse_args()
    finally:
        sys.argv = old_argv
    bundle = read_manifest(manifest)
    stage_args.hops = sorted({int(row["hop_count"]) for row in bundle})
    stage_args.output_dir.mkdir(parents=True, exist_ok=True)
    module.base.write_jsonl(stage_args.output_dir / "oracle_questions.jsonl", bundle)
    counts = Counter(int(row["hop_count"]) for row in bundle)
    validation = {
        "questions": len(bundle),
        "hops": stage_args.hops,
        "questions_by_hop": {str(hop): counts[hop] for hop in stage_args.hops},
        "source_manifest": str(manifest.resolve()),
        "path_type": "single_relation_chain",
        "hop_counts_match_statement_counts": all(
            int(item["hop_count"]) == len(item["gold_statement_ids"]) for item in bundle
        ),
    }
    module.base.write_json(stage_args.output_dir / "bundle_validation.json", validation)
    if stage_args.build_only:
        module.base.write_json(
            stage_args.output_dir / "metrics.json",
            {"status": "build_only", "llm_accuracy": None, **validation},
        )
        return
    predictions, report = module.evaluate(bundle, stage_args)
    module.base.write_jsonl(stage_args.output_dir / "predictions.jsonl", predictions)
    module.base.write_json(stage_args.output_dir / "metrics.json", report)


def correct_e3_metadata(output_dir: Path, counts: dict[int, int]) -> None:
    for filename in ("configuration.json", "metrics.json"):
        path = output_dir / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        target = payload.get("configuration", payload)
        if isinstance(target, dict):
            target["questions"] = sum(counts.values())
            target.pop("questions_per_hop", None)
            target["questions_by_hop"] = {str(hop): count for hop, count in counts.items()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args, forwarded = parse_args()
    require_key(args.build_only)
    os.environ["OPENAI_REASONING_EFFORT"] = args.reasoning_effort

    module = load_module(TARGETS[args.stage], f"gpt54_kqapro_{args.stage}")
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
        if not has_flag(forwarded, "--selector-max-tokens"):
            forwarded += ["--selector-max-tokens", "512"]
        if not has_flag(forwarded, "--answer-max-tokens"):
            forwarded += ["--answer-max-tokens", "256"]
    if args.stage == "e3" and not has_flag(forwarded, "--selector-api"):
        forwarded += ["--selector-api", "openai-compatible"]
    if args.manifest is not None and args.stage in {"e0", "e2", "e3"}:
        if not has_flag(forwarded, "--questions"):
            forwarded += ["--questions", str(args.manifest)]
    if args.stage == "e2":
        e3_output = args.output_root / "e3"
        dependencies = {
            "--candidate-banks": e3_output / "candidate_banks.jsonl",
            "--selection-results": e3_output / "selection_results.jsonl",
            "--e3-configuration": e3_output / "configuration.json",
        }
        for flag, path in dependencies.items():
            if not has_flag(forwarded, flag):
                forwarded += [flag, str(path)]

    counts: dict[int, int] | None = None
    if args.stage == "e3":
        manifest_path = args.manifest
        if manifest_path is None:
            manifest_path = Path(
                option_value(forwarded, "--questions")
                or PROCESSING
                / "knowledge_representation_processing"
                / "output"
                / "oracle_hops_15"
                / "oracle_questions.jsonl"
            )
        counts = configure_e3_manifest(module, manifest_path)

    if args.stage == "e1" and args.manifest is not None:
        run_e1_manifest(module, args.manifest, forwarded)
    else:
        invoke(module, forwarded)
    if counts is not None:
        correct_e3_metadata(stage_output, counts)
    annotate_output(stage_output, args.model, args.api_base, args.reasoning_effort)


if __name__ == "__main__":
    main()
