"""
Unit & Integration Tests for T5 Evaluation Harness and Baselines.
Covers:
- Metric calculations on small hand-built analytical examples
- Majority-intent prediction logic
- Always-escalate prediction logic
- Missing, invalid, and malformed gold labels
- Duplicate thread IDs and row count validation
- End-to-end evaluation execution on data/gold/gold_eval_200.jsonl
"""

import unittest
import copy
from pathlib import Path

from src.evaluation.metrics import (
    compute_accuracy,
    compute_multiclass_metrics,
    compute_binary_escalation_metrics,
    compute_confusion_matrix,
)
from src.evaluation.baselines import (
    MajorityIntentBaseline,
    AlwaysEscalateBaseline,
    TrivialBaselineAgent,
)
from src.evaluation.validator import (
    validate_gold_records,
    GoldValidationError,
    load_and_validate_gold,
    ALLOWED_INTENTS,
)
from scripts.eval_gold import run_evaluation


class TestMetricsCalculations(unittest.TestCase):
    """Verifies metric mathematics against exact hand-calculated analytical cases."""

    def test_hand_built_multiclass_metrics(self):
        # 3 classes: A, B, C (2 of each in true labels)
        y_true = ["A", "A", "B", "B", "C", "C"]
        y_pred = ["A", "B", "B", "B", "C", "A"]
        classes = ["A", "B", "C"]

        metrics = compute_multiclass_metrics(y_true, y_pred, labels=classes)

        # Accuracy: 4/6 correct = 0.6667
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6, places=4)

        # Class A: TP=1 (idx 0), FP=1 (idx 5), FN=1 (idx 1)
        # Precision = 1/2 = 0.5, Recall = 1/2 = 0.5, F1 = 0.5
        self.assertAlmostEqual(metrics["per_class"]["A"]["precision"], 0.5, places=4)
        self.assertAlmostEqual(metrics["per_class"]["A"]["recall"], 0.5, places=4)
        self.assertAlmostEqual(metrics["per_class"]["A"]["f1"], 0.5, places=4)
        self.assertEqual(metrics["per_class"]["A"]["support"], 2)

        # Class B: TP=2 (idx 2, 3), FP=1 (idx 1), FN=0
        # Precision = 2/3 = 0.6667, Recall = 2/2 = 1.0, F1 = 2*(2/3)*1 / (2/3 + 1) = 0.8
        self.assertAlmostEqual(metrics["per_class"]["B"]["precision"], 2 / 3, places=4)
        self.assertAlmostEqual(metrics["per_class"]["B"]["recall"], 1.0, places=4)
        self.assertAlmostEqual(metrics["per_class"]["B"]["f1"], 0.8, places=4)
        self.assertEqual(metrics["per_class"]["B"]["support"], 2)

        # Class C: TP=1 (idx 4), FP=0, FN=1 (idx 5)
        # Precision = 1/1 = 1.0, Recall = 1/2 = 0.5, F1 = 2*1*0.5 / 1.5 = 2/3 = 0.6667
        self.assertAlmostEqual(metrics["per_class"]["C"]["precision"], 1.0, places=4)
        self.assertAlmostEqual(metrics["per_class"]["C"]["recall"], 0.5, places=4)
        self.assertAlmostEqual(metrics["per_class"]["C"]["f1"], 2 / 3, places=4)
        self.assertEqual(metrics["per_class"]["C"]["support"], 2)

        # Macro-F1 = (0.5 + 0.8 + 2/3) / 3 = (1.3 + 0.66667) / 3 = 1.96667 / 3 = 0.6556
        expected_macro_f1 = (0.5 + 0.8 + (2 / 3)) / 3
        self.assertAlmostEqual(metrics["macro_f1"], expected_macro_f1, places=4)

        # Check confusion matrix layout: rows=actual, cols=predicted
        cm = metrics["confusion_matrix"]
        # Row A: 1 A, 1 B, 0 C
        self.assertEqual(cm[0], [1, 1, 0])
        # Row B: 0 A, 2 B, 0 C
        self.assertEqual(cm[1], [0, 2, 0])
        # Row C: 1 A, 0 B, 1 C
        self.assertEqual(cm[2], [1, 0, 1])

    def test_hand_built_binary_escalation_metrics(self):
        y_true = [True, True, False, False]
        y_pred = [True, False, True, False]

        # TP: item 0 (True, True) -> 1
        # FN: item 1 (True, False) -> 1
        # FP: item 2 (False, True) -> 1
        # TN: item 3 (False, False) -> 1
        metrics = compute_binary_escalation_metrics(y_true, y_pred)

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 0.5, places=4)
        self.assertAlmostEqual(metrics["precision"], 0.5, places=4)
        self.assertAlmostEqual(metrics["recall"], 0.5, places=4)
        self.assertAlmostEqual(metrics["f1"], 0.5, places=4)


class TestBaselines(unittest.TestCase):
    """Verifies behavior of dummy baselines."""

    def test_majority_intent_baseline(self):
        toy_records = [
            {"thread_id": "t1", "intent": "battery_power_issue"},
            {"thread_id": "t2", "intent": "battery_power_issue"},
            {"thread_id": "t3", "intent": "apps_feature_howto"},
        ]
        baseline = MajorityIntentBaseline().fit(toy_records)
        self.assertEqual(baseline.majority_intent, "battery_power_issue")

        preds = baseline.predict(toy_records)
        self.assertEqual(len(preds), 3)
        for p in preds:
            self.assertEqual(p["predicted_intent"], "battery_power_issue")
            self.assertEqual(p["baseline_name"], "majority_intent")

    def test_always_escalate_baseline(self):
        toy_records = [
            {"thread_id": "t1", "escalate": False},
            {"thread_id": "t2", "escalate": True},
        ]
        baseline = AlwaysEscalateBaseline()
        preds = baseline.predict(toy_records)
        self.assertEqual(len(preds), 2)
        for p in preds:
            self.assertTrue(p["predicted_escalate"])
            self.assertEqual(p["baseline_name"], "always_escalate")

    def test_trivial_baseline_agent(self):
        toy_records = [
            {"thread_id": "t1", "customer_text": "Need help", "intent": "battery_power_issue", "escalate": True, "escalate_reason": "physical_hardware_safety"},
            {"thread_id": "t2", "customer_text": "Battery bad", "intent": "battery_power_issue", "escalate": True, "escalate_reason": "physical_hardware_safety"},
            {"thread_id": "t3", "customer_text": "How to update", "intent": "apps_feature_howto", "escalate": False, "escalate_reason": None},
        ]
        agent = TrivialBaselineAgent().fit(toy_records)
        self.assertEqual(agent.majority_intent, "battery_power_issue")
        self.assertTrue(agent.escalate)
        self.assertEqual(agent.escalate_reason, "physical_hardware_safety")

        preds = agent.predict(toy_records)
        self.assertEqual(len(preds), 3)
        for p in preds:
            self.assertEqual(p["predicted_intent"], "battery_power_issue")
            self.assertTrue(p["predicted_escalate"])
            self.assertEqual(p["predicted_escalate_reason"], "physical_hardware_safety")
            self.assertIn("https://t.co/GDrqU22YpT", p["predicted_reply"])
            self.assertEqual(p["baseline_name"], "trivial_baseline_agent")

    def test_hand_built_lexical_reply_metrics(self):
        from src.evaluation.metrics import (
            tokenize_text,
            compute_ngram_overlap,
            compute_rouge_l,
            compute_bleu_1,
            compute_reply_lexical_metrics
        )
        # Known pair
        ref = "we are here to help please dm us"
        hyp = "we want to help send us a dm"
        ref_toks = tokenize_text(ref)
        hyp_toks = tokenize_text(hyp)

        # Unigrams: 5 common words (we, to, help, dm, us) out of 8 words in each
        p1, r1, f1 = compute_ngram_overlap(ref_toks, hyp_toks, n=1)
        self.assertAlmostEqual(p1, 5 / 8, places=4)
        self.assertAlmostEqual(r1, 5 / 8, places=4)
        self.assertAlmostEqual(f1, 5 / 8, places=4)

        # Bigrams: only ('to', 'help') is common out of 7 bigrams in each
        p2, r2, f2 = compute_ngram_overlap(ref_toks, hyp_toks, n=2)
        self.assertAlmostEqual(p2, 1 / 7, places=4)
        self.assertAlmostEqual(r2, 1 / 7, places=4)

        # ROUGE-L: LCS is 4 ('we', 'to', 'help', 'dm') or ('we', 'to', 'help', 'us')
        pl, rl, fl = compute_rouge_l(ref_toks, hyp_toks)
        self.assertAlmostEqual(pl, 4 / 8, places=4)
        self.assertAlmostEqual(rl, 4 / 8, places=4)

        # Overall batch metrics
        batch_metrics = compute_reply_lexical_metrics([ref], [hyp])
        self.assertAlmostEqual(batch_metrics["rouge_1_f1"], 0.625, places=3)
        self.assertAlmostEqual(batch_metrics["rouge_2_f1"], 1 / 7, places=3)
        self.assertAlmostEqual(batch_metrics["rouge_l_f1"], 0.5, places=3)
        self.assertEqual(batch_metrics["num_evaluated_replies"], 1)


class TestGoldDatasetValidation(unittest.TestCase):
    """Verifies that validator catches all classes of invalid gold records."""

    def setUp(self):
        # Create a valid mock batch of 2 records for testing validation errors
        self.valid_mock = [
            {
                "thread_id": "mock_001",
                "customer_tweet_id": "111",
                "customer_text": "Battery dying",
                "brand_text": "How can we help?",
                "intent": "battery_power_issue",
                "escalate": False,
                "escalate_reason": None,
                "notes": "Valid note"
            },
            {
                "thread_id": "mock_002",
                "customer_tweet_id": "222",
                "customer_text": "Password locked",
                "brand_text": "Check DM",
                "intent": "account_access_security",
                "escalate": True,
                "escalate_reason": "account_security_credentials",
                "notes": "Valid note"
            }
        ]

    def test_catches_invalid_intent(self):
        invalid_records = copy.deepcopy(self.valid_mock)
        invalid_records[0]["intent"] = "unsupported_non_canonical_intent"
        with self.assertRaises(GoldValidationError) as ctx:
            validate_gold_records(invalid_records, expected_count=2)
        self.assertIn("invalid intent", str(ctx.exception))

    def test_catches_missing_intent(self):
        invalid_records = copy.deepcopy(self.valid_mock)
        invalid_records[0]["intent"] = None
        with self.assertRaises(GoldValidationError) as ctx:
            validate_gold_records(invalid_records, expected_count=2)
        self.assertIn("missing intent label", str(ctx.exception))

    def test_catches_duplicate_thread_ids(self):
        dup_records = copy.deepcopy(self.valid_mock)
        dup_records[1]["thread_id"] = "mock_001"  # duplicate ID
        with self.assertRaises(GoldValidationError) as ctx:
            validate_gold_records(dup_records, expected_count=2)
        self.assertIn("duplicate thread IDs", str(ctx.exception))

    def test_catches_wrong_row_count(self):
        records = copy.deepcopy(self.valid_mock)
        with self.assertRaises(GoldValidationError) as ctx:
            # Expected 3 but provided 2
            validate_gold_records(records, expected_count=3)
        self.assertIn("Expected exactly 3 records", str(ctx.exception))

    def test_catches_non_boolean_escalate(self):
        invalid_records = copy.deepcopy(self.valid_mock)
        invalid_records[0]["escalate"] = "False"  # string instead of boolean
        with self.assertRaises(GoldValidationError) as ctx:
            validate_gold_records(invalid_records, expected_count=2)
        self.assertIn("'escalate' must be boolean", str(ctx.exception))

    def test_catches_escalate_true_without_reason(self):
        invalid_records = copy.deepcopy(self.valid_mock)
        invalid_records[1]["escalate_reason"] = None
        with self.assertRaises(GoldValidationError) as ctx:
            validate_gold_records(invalid_records, expected_count=2)
        self.assertIn("escalate=True requires an escalate_reason", str(ctx.exception))

    def test_catches_escalate_false_with_reason(self):
        invalid_records = copy.deepcopy(self.valid_mock)
        invalid_records[0]["escalate_reason"] = "channel_transition"
        with self.assertRaises(GoldValidationError) as ctx:
            validate_gold_records(invalid_records, expected_count=2)
        self.assertIn("escalate=False requires null/empty escalate_reason", str(ctx.exception))


class TestEvaluationEndToEnd(unittest.TestCase):
    """Verifies end-to-end evaluation harness on actual gold_eval_200.jsonl."""

    def test_actual_gold_evaluation_run(self):
        gold_file = Path("data/gold/gold_eval_200.jsonl")
        holdout_file = Path("data/gold/index_holdout_ids.txt")
        reports_dir = Path("reports")

        self.assertTrue(gold_file.exists(), "gold_eval_200.jsonl must exist")
        self.assertTrue(holdout_file.exists(), "index_holdout_ids.txt must exist")

        results = run_evaluation(gold_file, holdout_file, reports_dir)

        # Assert metadata
        self.assertEqual(results["metadata"]["num_samples"], 200)

        # Assert Trivial Baseline Agent (T6)
        self.assertIn("trivial_baseline_agent", results)
        tb = results["trivial_baseline_agent"]
        self.assertEqual(tb["majority_intent"], "other")
        self.assertTrue(tb["escalate_decision"])
        self.assertEqual(tb["escalate_reason"], "channel_transition")
        self.assertAlmostEqual(tb["intent_metrics"]["accuracy"], 0.16, places=2)
        self.assertAlmostEqual(tb["intent_metrics"]["macro_f1"], 0.0276, places=4)
        self.assertAlmostEqual(tb["escalation_metrics"]["accuracy"], 0.515, places=3)
        self.assertAlmostEqual(tb["escalation_metrics"]["f1"], 0.6799, places=4)
        self.assertGreater(tb["reply_metrics"]["rouge_1_f1"], 0.0)
        self.assertGreater(tb["reply_metrics"]["rouge_l_f1"], 0.0)

        # Assert predictions file was generated with 200 items
        pred_file = reports_dir / "eval_results_trivial.jsonl"
        self.assertTrue(pred_file.exists())
        import json
        with open(pred_file, "r", encoding="utf-8") as f:
            pred_lines = [json.loads(line) for line in f]
        self.assertEqual(len(pred_lines), 200)
        self.assertIn("predicted_reply", pred_lines[0])
        self.assertIn("gold_reply", pred_lines[0])

        # Assert Component Baseline 1 (Majority Intent)
        if "majority_intent_baseline" in results:
            b1 = results["majority_intent_baseline"]
            self.assertEqual(b1["predicted_class"], "other")
            self.assertAlmostEqual(b1["accuracy"], 0.16, places=2)
            self.assertAlmostEqual(b1["macro_f1"], 0.0276, places=4)

        # Assert Component Baseline 2 (Always Escalate)
        if "always_escalate_baseline" in results:
            b2 = results["always_escalate_baseline"]
            self.assertEqual(b2["tp"], 103)
            self.assertEqual(b2["fp"], 97)
            self.assertEqual(b2["tn"], 0)
            self.assertEqual(b2["fn"], 0)
            self.assertAlmostEqual(b2["accuracy"], 0.515, places=3)
            self.assertAlmostEqual(b2["recall"], 1.0, places=4)
            self.assertAlmostEqual(b2["precision"], 0.515, places=3)
            self.assertAlmostEqual(b2["f1"], 0.6799, places=4)


if __name__ == "__main__":
    unittest.main()
