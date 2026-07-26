#!/usr/bin/env python3
"""Plan or execute one frozen formal E0/E1/E3/E2 experiment run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
PROCESSING = PROJECT / "processing"
INPUT_ROOT = HERE / "output" / "inputs"
RUN_ROOT = HERE / "output" / "runs"
STAGE_ORDER = ("e0", "e1", "e3", "e2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("kqapro", "grailqa"), required=True)
    parser.add_argument(
        "--backend", choices=("openai", "ollama", "deepinfra"), required=True
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=(*STAGE_ORDER, "all"),
        default=["all"],
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run model calls. Without this flag only a frozen plan is written.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "changed_paths": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "changed_paths": []}


def validate_inputs(dataset: str, input_dir: Path) -> dict[str, Any]:
    questions = (
        input_dir / "oracle_questions.jsonl"
        if dataset == "kqapro"
        else input_dir / "selected_questions.jsonl"
    )
    required = [questions]
    if dataset == "grailqa":
        required += [
            input_dir / "mini_kb.jsonl",
            input_dir / "query_results.jsonl",
            input_dir / "answer_alias_cache.json",
        ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    rows = read_jsonl(questions)
    counts = Counter(int(row["hop_count"]) for row in rows)
    expected = {1: 100, 2: 100, 3: 60}
    if dict(counts) != expected:
        raise RuntimeError(f"Expected formal counts {expected}, found {dict(counts)}")
    ids = [
        str(row.get("question_id", row.get("qid")))
        for row in rows
    ]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Formal question IDs are not unique")
    sufficient_gold_counts: list[int] | None = None
    max_answer_count: int | None = None
    if dataset == "grailqa":
        missing_sufficient = [
            str(row["question_id"])
            for row in rows
            if not row.get("sufficient_gold_statement_ids")
        ]
        if missing_sufficient:
            raise RuntimeError(
                "Formal GrailQA questions miss sufficient Gold paths: "
                f"{missing_sufficient[:20]}"
            )
        non_subsets = [
            str(row["question_id"])
            for row in rows
            if not set(row["sufficient_gold_statement_ids"])
            <= set(row["gold_statement_ids"])
        ]
        if non_subsets:
            raise RuntimeError(
                "Sufficient Gold is not a subset of materialized Gold: "
                f"{non_subsets[:20]}"
            )
        sufficient_gold_counts = [
            len(row["sufficient_gold_statement_ids"]) for row in rows
        ]
        if max(sufficient_gold_counts) > 50:
            raise RuntimeError(
                "Formal GrailQA sufficient Gold exceeds selector top-k 50: "
                f"{max(sufficient_gold_counts)}"
            )
        max_answer_count = max(len(row["answer_ids"]) for row in rows)
        if max_answer_count > 128:
            raise RuntimeError(
                "Formal GrailQA answer count exceeds structured-output limit 128: "
                f"{max_answer_count}"
            )
    return {
        "questions": str(questions.resolve()),
        "questions_sha256": sha256_file(questions),
        "questions_by_hop": {str(hop): counts[hop] for hop in sorted(counts)},
        "question_count": len(rows),
        "mini_kb_sha256": (
            sha256_file(input_dir / "mini_kb.jsonl")
            if dataset == "grailqa"
            else None
        ),
        "query_results_sha256": (
            sha256_file(input_dir / "query_results.jsonl")
            if dataset == "grailqa"
            else None
        ),
        "alias_cache_sha256": (
            sha256_file(input_dir / "answer_alias_cache.json")
            if dataset == "grailqa"
            else None
        ),
        "sufficient_gold_statement_count_min": (
            min(sufficient_gold_counts) if sufficient_gold_counts else None
        ),
        "sufficient_gold_statement_count_max": (
            max(sufficient_gold_counts) if sufficient_gold_counts else None
        ),
        "maximum_answer_count": max_answer_count,
    }


def stage_commands(
    args: argparse.Namespace,
    stages: list[str],
    model: str,
    base_url: str,
    input_dir: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    if args.dataset == "kqapro":
        manifest = input_dir / "oracle_questions.jsonl"
        runners = {
            "openai": HERE / "run_kqapro_openai.py",
            "ollama": HERE / "run_kqapro_ollama.py",
            "deepinfra": HERE / "run_kqapro_deepinfra.py",
        }
        runner = runners[args.backend]
        for stage in stages:
            command = [
                sys.executable,
                str(runner),
                stage,
                "--manifest",
                str(manifest),
                "--model",
                model,
                "--output-root",
                str(output_root),
            ]
            command += [
                "--base-url" if args.backend == "ollama" else "--api-base",
                base_url,
            ]
            commands.append({"stage": stage, "command": command})
        return commands

    questions = input_dir / "selected_questions.jsonl"
    kb = input_dir / "mini_kb.jsonl"
    query_results = input_dir / "query_results.jsonl"
    alias_cache = input_dir / "answer_alias_cache.json"
    datasets = [
        PROJECT
        / "GrailQA_v1.0 (1)"
        / "GrailQA_v1.0"
        / "grailqa_v1.0_train.json",
        PROJECT
        / "GrailQA_v1.0 (1)"
        / "GrailQA_v1.0"
        / "grailqa_v1.0_dev.json",
    ]
    targets = {
        "e0": PROCESSING / "grailqa_e0_processing" / "run_e0_closed_book.py",
        "e1": PROCESSING / "grailqa_e1_gold_only_processing" / "run_e1_gold_only.py",
        "e2": PROCESSING / "grailqa_e2_forced_gold_processing" / "run_e2_forced_gold.py",
        "e3": (
            PROCESSING
            / "grailqa_e3_manual_llm_retrieval_processing"
            / "run_e3_manual_retrieval.py"
        ),
    }
    openai_runner = (
        PROCESSING / "openai_gpt54_experiments" / "run_grailqa_openai.py"
    )
    deepinfra_runner = HERE / "run_grailqa_deepinfra.py"
    for stage in stages:
        if args.backend in {"openai", "deepinfra"}:
            command = [
                sys.executable,
                str(openai_runner if args.backend == "openai" else deepinfra_runner),
                stage,
                "--model",
                model,
                "--api-base",
                base_url,
                "--output-root",
                str(output_root),
            ]
        else:
            command = [
                sys.executable,
                str(targets[stage]),
                "--model",
                model,
                "--base-url",
                base_url,
                "--output-dir",
                str(output_root / stage),
            ]
        command += ["--questions", str(questions), "--alias-cache", str(alias_cache)]
        if stage in {"e1", "e2", "e3"}:
            command += ["--kb", str(kb)]
        if stage == "e1":
            command += ["--gold-context", "sufficient"]
        if stage in {"e2", "e3"}:
            command += [
                "--query-results",
                str(query_results),
                "--datasets",
                *[str(path) for path in datasets],
                "--noise-statements",
                "100",
                "--top-k",
                "50",
                "--seed",
                "42",
            ]
        if stage == "e2":
            command += [
                "--e3-selection-results",
                str(output_root / "e3" / "selection_results.jsonl"),
            ]
        commands.append({"stage": stage, "command": command})
    return commands


def main() -> None:
    args = parse_args()
    stages = list(STAGE_ORDER) if "all" in args.stages else list(args.stages)
    if len(stages) != len(set(stages)):
        raise ValueError("Duplicate stages requested")
    stages.sort(key=STAGE_ORDER.index)
    default_models = {
        "openai": "gpt-5.4-2026-03-05",
        "ollama": "qwen3-4b-16k",
        "deepinfra": (
            "Qwen/Qwen3-14B:"
            "01b0452aeedc42eaf4a73d87654a8129d6a436f0"
        ),
    }
    default_base_urls = {
        "openai": "https://api.openai.com/v1",
        "ollama": "http://127.0.0.1:11434/v1",
        "deepinfra": "https://api.deepinfra.com/v1/openai",
    }
    model = args.model or default_models[args.backend]
    base_url = args.base_url or default_base_urls[args.backend]
    input_dir = args.input_dir or (
        INPUT_ROOT
        / ("kqapro_100_100_60" if args.dataset == "kqapro" else "grailqa_100_100_60")
    )
    output_root = args.output_root or (
        RUN_ROOT / args.dataset / args.backend / safe_name(model)
    )
    validation = validate_inputs(args.dataset, input_dir)
    if "e2" in stages and "e3" not in stages:
        selection_path = output_root / "e3" / "selection_results.jsonl"
        if not selection_path.exists():
            raise FileNotFoundError(
                f"E2 requires the same model's saved E3 selections: {selection_path}"
            )

    commands = stage_commands(
        args, stages, model, base_url, input_dir, output_root
    )
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "planned" if not args.execute else "running",
        "dataset": args.dataset,
        "backend": args.backend,
        "model": model,
        "base_url": base_url,
        "stages": stages,
        "stage_order": list(STAGE_ORDER),
        "formal_input": validation,
        "output_root": str(output_root.resolve()),
        "controls": {
            "representations": ["document", "triple", "graph"],
            "e2_reuses_same_model_e3_selection": True,
            "noise_seed": 42,
            "kqapro_candidate_statements": 100,
            "kqapro_top_k": 20,
            "grailqa_noise_statements": 100,
            "grailqa_top_k": 50,
            "grailqa_max_sufficient_gold_statements": 50,
        },
        "git": git_state(),
        "commands": commands,
    }
    plan_path = output_root / "run_plan.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        print("\nPlan only. Add --execute to start model calls.")
        return

    plan["completed_stages"] = []
    try:
        for row in commands:
            print(f"\n=== {row['stage'].upper()} ===", flush=True)
            subprocess.run(
                row["command"], cwd=PROJECT, check=True, env=os.environ.copy()
            )
            plan["completed_stages"].append(row["stage"])
            plan_path.write_text(
                json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except (OSError, subprocess.CalledProcessError) as error:
        plan["status"] = "failed"
        plan["failed_stage"] = row["stage"]
        plan["error"] = str(error)
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    plan["status"] = "completed"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
