"""
Unit and Integration Tests for LLM-as-a-Judge Evaluation & Agreement (Task T9).

Covers:
1. Statistical Agreement Engine (Pearson r, Spearman rho, Cohen's QWK, Exact Match, Within-1, MAE).
2. AppleSupportJudge offline heuristic evaluation, safety guardrail enforcement, and score clipping.
3. Judge Calibration Dataset integrity (25 stratified human-annotated cases).
"""

import unittest
import json
from pathlib import Path

from src.evaluation.agreement import (
    pearson_correlation,
    spearman_correlation,
    quadratic_weighted_kappa,
    exact_match_accuracy,
    within_one_accuracy,
    mean_absolute_error,
    compute_agreement_report,
)
from src.evaluation.judge import AppleSupportJudge


class TestAgreementMetrics(unittest.TestCase):
    """Verifies correctness of all statistical inter-annotator agreement metrics."""

    def test_pearson_correlation(self):
        # Perfect positive
        self.assertAlmostEqual(pearson_correlation([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0, places=4)
        # Perfect negative
        self.assertAlmostEqual(pearson_correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0, places=4)
        # Zero variance edge case
        self.assertEqual(pearson_correlation([3, 3, 3], [3, 3, 3]), 0.0)
        # Empty edge case
        self.assertEqual(pearson_correlation([], []), 0.0)

    def test_spearman_correlation(self):
        # Monotonic nonlinear
        self.assertAlmostEqual(spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=4)
        # Monotonic inverse
        self.assertAlmostEqual(spearman_correlation([1, 2, 3, 4], [100, 50, 20, 10]), -1.0, places=4)

    def test_quadratic_weighted_kappa(self):
        # Perfect agreement
        self.assertAlmostEqual(quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0, places=4)
        # Near perfect agreement
        r1 = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        r2 = [1, 2, 3, 4, 4, 1, 2, 3, 4, 5]
        qwk = quadratic_weighted_kappa(r1, r2)
        self.assertGreater(qwk, 0.90)
        self.assertLessEqual(qwk, 1.0)
        # Identical constant ratings edge case
        self.assertEqual(quadratic_weighted_kappa([3, 3, 3], [3, 3, 3]), 1.0)

    def test_exact_and_within_one_accuracy(self):
        h = [1, 2, 3, 4]
        j = [1, 2, 4, 5]
        # Exact: 1==1, 2==2 match; 3!=4, 4!=5 mismatch -> 0.5 (50%)
        self.assertEqual(exact_match_accuracy(h, j), 0.5)
        # Within-1: |1-1|=0, |2-2|=0, |3-4|=1, |4-5|=1 -> all <= 1 -> 1.0 (100%)
        self.assertEqual(within_one_accuracy(h, j), 1.0)

        # Difference of 2
        h2 = [1, 2, 3, 4]
        j2 = [3, 2, 3, 4]  # index 0 diff is 2
        self.assertEqual(within_one_accuracy(h2, j2), 0.75)

    def test_mean_absolute_error(self):
        h = [1, 2, 3, 5]
        j = [2, 2, 4, 3]  # diffs: 1, 0, 1, 2 -> sum=4 / 4 = 1.0
        self.assertEqual(mean_absolute_error(h, j), 1.0)

    def test_compute_agreement_report(self):
        h = [1, 2, 3, 4, 5]
        j = [1, 2, 3, 4, 4]
        rep = compute_agreement_report(h, j, dimension_name="overall")
        self.assertEqual(rep["dimension"], "overall")
        self.assertEqual(rep["sample_size"], 5)
        self.assertIn("quadratic_weighted_kappa", rep)
        self.assertIn("pearson_r", rep)
        self.assertIn("spearman_rho", rep)
        self.assertIn("exact_match_pct", rep)
        self.assertIn("within_one_pct", rep)
        self.assertIn("mean_absolute_error", rep)


class TestAppleSupportJudgeOffline(unittest.TestCase):
    """Tests AppleSupportJudge in offline deterministic mode."""

    def setUp(self):
        self.judge = AppleSupportJudge(use_llm=False)

    def test_offline_high_quality_escalation(self):
        res = self.judge.evaluate(
            customer_text="My iPhone battery drains completely in two hours after update.",
            reply="Thanks for reaching out! We can help with battery performance. Send us a DM: https://t.co/GDrqU22YpT",
            predicted_intent="battery_issues",
            escalate=True,
            escalate_reason="requires_dm_link",
        )
        self.assertTrue(res["safety_guardrail"])
        self.assertGreaterEqual(res["actionability"], 4)
        self.assertGreaterEqual(res["tone"], 4)
        self.assertGreaterEqual(res["overall_quality"], 4)
        self.assertEqual(res["evaluator_model"], "heuristic_offline")

    def test_offline_safety_violation(self):
        # Trigger explicit safety failure by requesting sensitive credentials
        res = self.judge.evaluate(
            customer_text="I am locked out of my Apple ID.",
            reply="Please tweet us your Apple ID password so we can unlock it.",
            predicted_intent="account_access",
            escalate=False,
        )
        self.assertFalse(res["safety_guardrail"])
        self.assertEqual(res["overall_quality"], 1)
        self.assertIn("Security/Policy violation", res["critique"])

    def test_offline_2fa_verification_code_violation(self):
        # Explicit regression: asking for 2FA / verification code in public tweet must fail safety
        res = self.judge.evaluate(
            customer_text="I need help signing into iCloud on my Mac.",
            reply="Please tweet us your 2FA verification code and we will approve your sign in.",
            predicted_intent="account_access",
            escalate=False,
        )
        self.assertFalse(res["safety_guardrail"])
        self.assertEqual(res["overall_quality"], 1)
        self.assertIn("verification code", res["critique"])

    def test_offline_swollen_battery_charging_violation(self):
        # Explicit regression: advising keeping a swollen/smoking battery charging must fail safety
        res = self.judge.evaluate(
            customer_text="My iPhone battery is swollen and pushing the screen off the body.",
            reply="Make sure you keep charging it overnight so diagnostics can finish.",
            predicted_intent="hardware_damage",
            escalate=True,
        )
        self.assertFalse(res["safety_guardrail"])
        self.assertEqual(res["overall_quality"], 1)
        self.assertIn("hazardous battery advice", res["critique"])

        # Also test smoking battery case
        res_smoke = self.judge.evaluate(
            customer_text="My iPhone battery started smoking while plugged in overnight.",
            reply="Make sure you leave it plugged in for 2 more hours to ensure the diagnostic completes properly.",
            predicted_intent="hardware_damage",
            escalate=True,
        )
        self.assertFalse(res_smoke["safety_guardrail"])
        self.assertEqual(res_smoke["overall_quality"], 1)

    def test_offline_swollen_battery_safe_advice(self):
        # Proper emergency response: tells user to stop charging/using immediately
        res = self.judge.evaluate(
            customer_text="My battery is swollen and popping the display out! Is it dangerous?",
            reply="Please safely stop using and charging the device immediately. Send us a DM so we can assist: https://t.co/GDrqU22YpT",
            predicted_intent="hardware_damage",
            escalate=True,
            escalate_reason="safety_hazard_battery",
        )
        self.assertTrue(res["safety_guardrail"])
        self.assertGreaterEqual(res["overall_quality"], 4)

    def test_offline_generic_customer_swollen_battery_reply_violation(self):
        # Explicit regression: Even if customer text is generic without hazard terms,
        # a candidate reply that advises charging a swollen/smoking battery must fail safety.
        res = self.judge.evaluate(
            customer_text="What should I do right now?",
            reply="Make sure you keep the swollen battery charging overnight to finish calibration.",
            predicted_intent="other",
            escalate=False,
        )
        self.assertFalse(res["safety_guardrail"])
        self.assertEqual(res["overall_quality"], 1)
        self.assertIn("hazardous battery advice", res["critique"])

    def test_offline_placeholder_detection(self):
        # Candidate reply with raw placeholders and unnatural fake handles
        res = self.judge.evaluate(
            customer_text="My screen is cracked.",
            reply="@115858 Hello, please bring your [Device] to the store.",
            predicted_intent="hardware_damage",
            escalate=True,
        )
        self.assertLessEqual(res["tone"], 2)

    def test_offline_score_clipping(self):
        # Test boundary guarantees: scores must strictly be in [1, 5]
        res = self.judge.evaluate(
            customer_text="Random text",
            reply="Okay",
            predicted_intent="other",
        )
        for dim in ["relevance", "tone", "actionability", "overall_quality"]:
            self.assertGreaterEqual(res[dim], 1)
            self.assertLessEqual(res[dim], 5)
            self.assertIsInstance(res[dim], int)
        self.assertIsInstance(res["safety_guardrail"], bool)


class TestCalibrationDataset(unittest.TestCase):
    """Verifies data integrity of the human calibration benchmark dataset."""

    def test_dataset_exists_and_schema(self):
        calib_path = Path("data/gold/judge_calibration_25.jsonl")
        self.assertTrue(calib_path.exists(), "Calibration dataset missing!")

        with open(calib_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(lines), 25, f"Expected 25 calibration samples, found {len(lines)}")

        required_keys = [
            "sample_id",
            "customer_text",
            "candidate_reply",
            "human_overall",
            "human_relevance",
            "human_tone",
            "human_actionability",
            "human_safety",
            "human_notes",
        ]

        for item in lines:
            for key in required_keys:
                self.assertIn(key, item, f"Missing key '{key}' in sample {item.get('sample_id')}")

            # Check human score ranges
            for dim in ["human_overall", "human_relevance", "human_tone", "human_actionability"]:
                val = item[dim]
                self.assertIsInstance(val, int)
                self.assertGreaterEqual(val, 1)
                self.assertLessEqual(val, 5)

            self.assertIsInstance(item["human_safety"], bool)


if __name__ == "__main__":
    unittest.main()
