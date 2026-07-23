"""Adaptive KQA Pro noise-bank construction for large frozen evaluations."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


LAST_AUDIT: dict[str, Any] = {}
HERE = Path(__file__).resolve().parent
DEFAULT_PILOT = (
    HERE.parent
    / "knowledge_representation_processing"
    / "output"
    / "oracle_hops_15"
    / "oracle_questions.jsonl"
)


def make_adaptive_builder(
    module: Any,
    minimum_unique_statements: int = 400,
) -> Callable[..., list[dict[str, Any]]]:
    """Return a deterministic builder that tolerates depleted per-hop pools."""

    def build(
        train: list[dict[str, Any]],
        kb: dict[str, Any],
        evaluation_questions: list[dict[str, Any]],
        background_per_hop: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        excluded = {int(item["train_index"]) for item in evaluation_questions}
        pilot_excluded: set[int] = set()
        if DEFAULT_PILOT.exists():
            with DEFAULT_PILOT.open(encoding="utf-8") as handle:
                pilot_excluded = {
                    int(json.loads(line)["train_index"])
                    for line in handle
                    if line.strip()
                }
            excluded |= pilot_excluded
        name_index = module.oracle_base.build_name_index(kb)
        statements: dict[str, dict[str, Any]] = {}
        sources: dict[str, list[str]] = defaultdict(list)
        accepted: dict[int, int] = {hop: 0 for hop in (1, 2, 3)}
        cursors: dict[int, int] = {hop: 0 for hop in (1, 2, 3)}
        pools: dict[int, list[tuple[int, dict[str, Any]]]] = {}

        for hop in (1, 2, 3):
            candidates = [
                (index, sample)
                for index, sample in enumerate(train)
                if index not in excluded
                and module.oracle_hops.is_linear_program(sample, hop)
                and sample.get("answer") in sample.get("choices", [])
            ]
            random.Random(seed + hop).shuffle(candidates)
            pools[hop] = candidates

        def accept_next(hop: int) -> bool:
            candidates = pools[hop]
            while cursors[hop] < len(candidates):
                train_index, sample = candidates[cursors[hop]]
                cursors[hop] += 1
                path = module.oracle_hops.trace_path(
                    sample, hop, kb, name_index
                )
                if path is None:
                    continue
                source_id = f"train_{train_index}"
                for statement in path:
                    statement_id = statement["statement_id"]
                    statements[statement_id] = statement
                    if source_id not in sources[statement_id] and len(
                        sources[statement_id]
                    ) < 20:
                        sources[statement_id].append(source_id)
                accepted[hop] += 1
                return True
            return False

        # Preserve the pilot policy where possible.
        for hop in (1, 2, 3):
            while accepted[hop] < background_per_hop and accept_next(hop):
                pass

        # A 60-question 3-hop evaluation depletes that background pool. Fill
        # the shared true-fact bank from the remaining non-evaluation paths,
        # cycling through hops so no single hop silently dominates.
        while len(statements) < minimum_unique_statements:
            progressed = False
            for hop in (1, 2, 3):
                if accept_next(hop):
                    progressed = True
                if len(statements) >= minimum_unique_statements:
                    break
            if not progressed:
                break
        if len(statements) < minimum_unique_statements:
            raise RuntimeError(
                "Formal noise pool exhausted: "
                f"{len(statements)} unique true statements found, "
                f"{minimum_unique_statements} required"
            )

        LAST_AUDIT.clear()
        LAST_AUDIT.update(
            {
                "policy": (
                    "up_to_background_per_hop_then_round_robin_non_evaluation_paths"
                ),
                "requested_background_paths_per_hop": background_per_hop,
                "accepted_background_paths_by_hop": {
                    str(hop): accepted[hop] for hop in (1, 2, 3)
                },
                "available_candidate_paths_by_hop": {
                    str(hop): len(pools[hop]) for hop in (1, 2, 3)
                },
                "minimum_unique_noise_statements": minimum_unique_statements,
                "actual_unique_noise_statements": len(statements),
                "evaluation_train_indices_excluded": len(evaluation_questions),
                "pilot_noise_source_indices_excluded": len(pilot_excluded),
                "total_train_indices_excluded": len(excluded),
            }
        )
        return [
            {
                "statement_id": statement_id,
                "statement": statements[statement_id],
                "source_question_ids": sorted(set(sources[statement_id])),
            }
            for statement_id in sorted(statements)
        ]

    return build


def annotate_output(output_dir: Path) -> None:
    """Add the formal background policy audit to generated JSON summaries."""
    if not LAST_AUDIT:
        return
    for filename in ("configuration.json", "metrics.json", "bundle_validation.json"):
        path = output_dir / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        target = payload.get("configuration", payload)
        if isinstance(target, dict):
            target["formal_noise_background"] = LAST_AUDIT
            target["noise_source"] = (
                "true gold-path statements from non-evaluation KQA Pro questions; "
                "adaptive per-hop background for formal scale"
            )
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
