#!/usr/bin/env python3
"""Unit tests for formal metric schema v2."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grailqa_e0_processing"))
sys.path.insert(0, str(HERE))

import run_e0_closed_book as e0  # noqa: E402


class AliasGroupScoringTests(unittest.TestCase):
    def test_single_token_a_is_preserved(self) -> None:
        self.assertEqual(e0.normalize_answer("A"), "a")
        result = e0.score_alias_answer_set(
            ["A"],
            ["A"],
            ["m.a"],
            {"m.a": set()},
        )
        self.assertTrue(result["alias_answer_set_em"])

    def test_same_label_mids_form_one_answer_group(self) -> None:
        result = e0.score_alias_answer_set(
            ["Same title"],
            ["Same title", "Same title"],
            ["m.1", "m.2"],
            {"m.1": {"same title"}, "m.2": {"same title"}},
        )
        self.assertTrue(result["alias_answer_set_em"])
        self.assertEqual(result["alias_gold_group_count"], 1)
        self.assertEqual(result["alias_matches"][0]["gold_mids"], ["m.1", "m.2"])

    def test_noncanonical_alias_matches_group(self) -> None:
        result = e0.score_alias_answer_set(
            ["SI"],
            ["International System of Units"],
            ["m.si"],
            {"m.si": {"si", "international system of units"}},
        )
        self.assertTrue(result["alias_answer_set_em"])

    def test_extra_prediction_is_not_exact_match(self) -> None:
        result = e0.score_alias_answer_set(
            ["SI", "metric"],
            ["International System of Units"],
            ["m.si"],
            {"m.si": {"si", "international system of units"}},
        )
        self.assertFalse(result["alias_answer_set_em"])
        self.assertEqual(result["alias_answer_precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
