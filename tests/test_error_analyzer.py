"""
Unit and Integration Tests for Error Analyzer & Failure Taxonomy (Task T10).

Covers:
1. Canonical failure mode classification rules.
2. Prediction error accounting (intent misclassifications, escalation false positives/negatives).
3. Machine-readable error dump and markdown report generation.
"""

import unittest
import json
import tempfile
from pathlib import Path

from src.evaluation.error_analyzer import (
    classify_failure_mode,
    analyze_predictions,
    dump_errors_and_summarize,
    FAILURE_MODES,
)


class TestErrorAnalyzer(unittest.TestCase):
    """Verifies failure mode categorization and error extraction logic."""

    def test_classify_failure_modes(self):
        # Mode 1: Update vs Performance
        r_mode1 = {
            "customer_text": "My phone is freezing constantly after updating to iOS 11.0.1",
            "gold_intent": "software_update_glitch",
            "predicted_intent": "performance_crash_freeze",
            "gold_escalate": False,
            "predicted_escalate": False,
        }
        mode, _ = classify_failure_mode(r_mode1)
        self.assertEqual(mode, "MODE_1_UPDATE_VS_PERFORMANCE")

        # Mode 2: Channel transition under-escalation (FN)
        r_mode2 = {
            "customer_text": "My photos are blurred in library",
            "gold_intent": "apps_feature_howto",
            "predicted_intent": "apps_feature_howto",
            "gold_escalate": True,
            "predicted_escalate": False,
            "gold_escalate_reason": "channel_transition",
        }
        mode, _ = classify_failure_mode(r_mode2)
        self.assertEqual(mode, "MODE_2_CHANNEL_TRANSITION_UNDER_ESCALATION")

        # Mode 3: How-to vs other
        r_mode3 = {
            "customer_text": "Can I trade in my watch band?",
            "gold_intent": "apps_feature_howto",
            "predicted_intent": "other",
            "gold_escalate": False,
            "predicted_escalate": False,
        }
        mode, _ = classify_failure_mode(r_mode3)
        self.assertEqual(mode, "MODE_3_HOWTO_VS_OTHER_SMEARING")

        # Mode 5: Multilingual
        r_mode5 = {
            "customer_text": "Voici mon problème avec l'application, aidez-moi s'il vous plaît",
            "gold_intent": "other",
            "predicted_intent": "other",
            "gold_escalate": False,
            "predicted_escalate": False,
        }
        mode, _ = classify_failure_mode(r_mode5)
        self.assertEqual(mode, "MODE_5_MULTILINGUAL_OR_MEDIA_CONTEXT")

    def test_analyze_predictions_and_dump(self):
        sample_preds = [
            {
                "thread_id": "test_01",
                "customer_text": "Phone freezes on lock screen",
                "gold_intent": "performance_crash_freeze",
                "predicted_intent": "performance_crash_freeze",
                "gold_escalate": False,
                "predicted_escalate": False,
                "gold_escalate_reason": None,
                "predicted_escalate_reason": None,
                "gold_reply": "Try restarting your iPhone.",
                "predicted_reply": "Try restarting your iPhone.",
            },
            {
                "thread_id": "test_02",
                "customer_text": "My battery drains in 2 hours after iOS 11 update",
                "gold_intent": "software_update_glitch",
                "predicted_intent": "battery_power_issue",
                "gold_escalate": True,
                "predicted_escalate": False,
                "gold_escalate_reason": "channel_transition",
                "predicted_escalate_reason": None,
                "gold_reply": "Please DM us so we can investigate: https://t.co/GDrqU22YpT",
                "predicted_reply": "Check your battery health in Settings.",
            },
            {
                "thread_id": "test_03",
                "customer_text": "How do I take a screenshot on iPhone X?",
                "gold_intent": "apps_feature_howto",
                "predicted_intent": "apps_feature_howto",
                "gold_escalate": False,
                "predicted_escalate": True,
                "gold_escalate_reason": None,
                "predicted_escalate_reason": "channel_transition",
                "gold_reply": "Press the side button and volume up simultaneously.",
                "predicted_reply": "Send us a DM: https://t.co/GDrqU22YpT",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            preds_file = tmppath / "test_preds.jsonl"
            with open(preds_file, "w", encoding="utf-8") as f:
                for p in sample_preds:
                    f.write(json.dumps(p) + "\n")

            out_dir = tmppath / "out"
            res = dump_errors_and_summarize(preds_file, out_dir)

            self.assertEqual(res["total_errors"], 2)
            self.assertEqual(res["intent_error_count"], 1)
            self.assertEqual(res["escalation_error_count"], 2)

            self.assertTrue(res["dump_path"].exists())
            self.assertTrue(res["summary_path"].exists())
            self.assertTrue(res["metrics_path"].exists())

            with open(res["dump_path"], "r", encoding="utf-8") as f:
                dumped_lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(dumped_lines), 2)


if __name__ == "__main__":
    unittest.main()
