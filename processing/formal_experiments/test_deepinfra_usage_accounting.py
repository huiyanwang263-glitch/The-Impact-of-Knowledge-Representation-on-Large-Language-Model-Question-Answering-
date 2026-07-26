#!/usr/bin/env python3
"""Tests for incremental versus condition-equivalent DeepInfra usage."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from deepinfra_run_metadata import aggregate_usage_views  # noqa: E402


def usage(response_id: str, tokens: int, cost: float) -> dict[str, object]:
    return {
        "response_id": response_id,
        "prompt_tokens": tokens - 1,
        "completion_tokens": 1,
        "total_tokens": tokens,
        "reasoning_tokens": 0,
        "estimated_cost": cost,
        "served_model": "Qwen/Qwen3-14B",
        "service_tier": "default",
        "finish_reason": "stop",
        "reasoning_content_present": False,
    }


class UsageAccountingTests(unittest.TestCase):
    def test_saved_e3_selector_is_excluded_from_incremental_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            row = {
                "selector_source": "saved_e3_selection",
                "selector_usage": usage("selector", 100, 0.01),
                "usage": usage("answer", 20, 0.002),
            }
            text = json.dumps(row) + "\n"
            (output_dir / "predictions.jsonl").write_text(text, encoding="utf-8")
            (output_dir / "checkpoint.jsonl").write_text(text, encoding="utf-8")
            views = aggregate_usage_views(output_dir)
        self.assertEqual(
            views["condition_equivalent"]["unique_model_calls"],
            2,
        )
        self.assertEqual(views["incremental_billed"]["unique_model_calls"], 1)
        self.assertEqual(views["incremental_billed"]["total_tokens"], 20)
        self.assertAlmostEqual(
            views["incremental_billed"]["estimated_cost_usd"],
            0.002,
        )

    def test_fresh_selector_is_incremental_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            row = {
                "selector_source": "fresh_selector",
                "selector_usage": usage("selector", 100, 0.01),
                "usage": usage("answer", 20, 0.002),
            }
            (output_dir / "predictions.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )
            views = aggregate_usage_views(output_dir)
        self.assertEqual(views["incremental_billed"]["unique_model_calls"], 2)


if __name__ == "__main__":
    unittest.main()
