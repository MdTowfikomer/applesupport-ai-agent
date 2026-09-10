"""
Unit and Integration Tests for AppleSupport AI Agent Pipeline (Task T8).

Covers:
1. HistoricalResolutionIndex: corpus loading, BM25 retrieval, and strict holdout isolation.
2. IntentClassifier: 10-class taxonomy and Codebook precedence rules.
3. EscalationRouter: Safety triggers, intent mandates, and valid reason codes.
4. ResponseDrafter: Grounded AppleSupport reply synthesis and DM escalation links.
5. AppleSupportAgent: 4-stage pipeline coordination and batch prediction.
"""

import unittest
from pathlib import Path

from src.pipeline.retriever import HistoricalResolutionIndex, tokenize
from src.pipeline.classifier import IntentClassifier
from src.pipeline.triage import EscalationRouter
from src.pipeline.drafter import ResponseDrafter
from src.pipeline.agent import AppleSupportAgent
from src.evaluation.validator import ALLOWED_INTENTS, ALLOWED_ESCALATE_REASONS


class TestRetriever(unittest.TestCase):
    """Verifies BM25 resolution retrieval and strict holdout leakage prevention."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = HistoricalResolutionIndex(
            subsample_path="data/subsample/applesupport_threads_5k.jsonl",
            holdout_ids_path="data/gold/index_holdout_ids.txt",
        )

    def test_holdout_isolation(self):
        """Asserts zero intersection between indexed corpus and gold holdout set."""
        self.assertEqual(len(self.retriever.holdout_ids), 200)
        self.assertEqual(len(self.retriever.corpus), 4800)
        corpus_ids = {doc["thread_id"] for doc in self.retriever.corpus}
        overlap = corpus_ids.intersection(self.retriever.holdout_ids)
        self.assertEqual(len(overlap), 0, f"Critical leakage! Overlapping IDs: {overlap}")

    def test_retrieve_structure_and_isolation(self):
        """Checks retrieved format and ensures results never return a holdout item."""
        results = self.retriever.retrieve("My iPhone battery dies in 2 hours", top_k=3)
        self.assertLessEqual(len(results), 3)
        self.assertGreater(len(results), 0)

        for match in results:
            self.assertIn("thread_id", match)
            self.assertIn("customer_text", match)
            self.assertIn("brand_text", match)
            self.assertIn("score", match)
            self.assertNotIn(match["thread_id"], self.retriever.holdout_ids)


class TestIntentClassifier(unittest.TestCase):
    """Verifies intent classification adherence to Codebook rules (offline rule-based)."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = IntentClassifier(use_llm=False)

    def test_canonical_intent_rules(self):
        cases = [
            ("My battery is draining so fast after charging to 100%", "battery_power_issue"),
            ("Twitter keeps crashing and my phone is totally frozen", "performance_crash_freeze"),
            ("Since updating to iOS 11.1, the autocorrect changes I to an A symbol", "software_update_glitch"),
            ("My iPhone won't connect to my home wifi network or bluetooth", "connectivity_network_issue"),
            ("I dropped my phone on concrete and the screen shattered", "hardware_physical_accessory"),
            ("My Apple ID is locked and verification code won't send", "account_access_security"),
            ("I was charged $9.99 for a subscription I cancelled, need a refund", "billing_purchases_subscriptions"),
            ("How do I create a new playlist in Apple Music?", "apps_feature_howto"),
            ("Nothing works this is the worst thing ever help me", "vague_complaint_unclear"),
            ("Hola no puedo usar mi iphone ayuda por favor", "other"),
        ]
        for query, expected_intent in cases:
            pred = self.classifier.classify(query)
            self.assertEqual(pred, expected_intent, f"Failed for query: '{query}' -> got {pred}")

    def test_codebook_precedence_rules(self):
        # Rule 3: account_access_security > billing > hardware > symptoms > vague
        # Query with account issue and crash symptom -> must be account
        query = "My phone froze and now my Apple ID password is locked"
        pred = self.classifier.classify(query)
        self.assertEqual(pred, "account_access_security")


class TestEscalationRouter(unittest.TestCase):
    """Verifies safety triggers, mandatory intent escalations, and reason codes."""

    @classmethod
    def setUpClass(cls):
        cls.router = EscalationRouter()

    def test_screenshot_only_trigger(self):
        esc, reason = self.router.triage("@AppleSupport", "other")
        self.assertTrue(esc)
        self.assertEqual(reason, "missing_context_screenshot")

    def test_mandatory_intent_escalations(self):
        # Account security intent -> must escalate with account_security_credentials
        esc, reason = self.router.triage("Locked out of my Apple ID", "account_access_security")
        self.assertTrue(esc)
        self.assertEqual(reason, "account_security_credentials")

        # Billing intent -> must escalate with financial_billing_transaction
        esc, reason = self.router.triage("Need refund for double charge", "billing_purchases_subscriptions")
        self.assertTrue(esc)
        self.assertEqual(reason, "financial_billing_transaction")

        # Hardware intent -> must escalate with physical_hardware_safety
        esc, reason = self.router.triage("Screen is cracked in half", "hardware_physical_accessory")
        self.assertTrue(esc)
        self.assertEqual(reason, "physical_hardware_safety")

    def test_safety_and_legal_triggers(self):
        # Safety / Battery swelling trigger
        esc, reason = self.router.triage("My iPhone battery is smoking and burning hot", "battery_power_issue")
        self.assertTrue(esc)
        self.assertEqual(reason, "physical_hardware_safety")

        # Legal dispute trigger
        esc, reason = self.router.triage("I am contacting my lawyer for a lawsuit against Apple", "other")
        self.assertTrue(esc)
        self.assertEqual(reason, "legal_regulatory_dispute")

    def test_routine_auto_handle(self):
        # Routine battery drain -> auto-handle (escalate = False)
        esc, reason = self.router.triage("My battery life is shorter than normal on iPhone 7", "battery_power_issue")
        self.assertFalse(esc)
        self.assertIsNone(reason)


class TestResponseDrafter(unittest.TestCase):
    """Verifies reply generation and tone requirements."""

    @classmethod
    def setUpClass(cls):
        cls.drafter = ResponseDrafter(use_llm=False)

    def test_escalated_reply_includes_dm_link(self):
        reply = self.drafter.draft(
            customer_text="My Apple ID is locked",
            intent="account_access_security",
            escalate=True,
            escalate_reason="account_security_credentials",
        )
        self.assertIn("https://t.co/GDrqU22YpT", reply)
        self.assertLess(len(reply), 280)

    def test_vague_reply_asks_clarification(self):
        reply = self.drafter.draft(
            customer_text="Help it is broken",
            intent="vague_complaint_unclear",
            escalate=False,
            escalate_reason=None,
        )
        self.assertIn("device model", reply.lower())
        self.assertIn("ios version", reply.lower())


class TestAppleSupportAgent(unittest.TestCase):
    """Verifies end-to-end agent orchestration across the 4-stage pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.agent = AppleSupportAgent(use_llm=False)

    def test_end_to_end_process(self):
        res = self.agent.process("My iPhone 8 battery drains from 100% to 0% in 1 hour")
        self.assertEqual(res["predicted_intent"], "battery_power_issue")
        self.assertFalse(res["predicted_escalate"])
        self.assertIsNone(res["predicted_escalate_reason"])
        self.assertTrue(len(res["predicted_reply"]) > 0)
        self.assertIsNotNone(res["retrieved_thread_id"])
        self.assertNotIn(res["retrieved_thread_id"], self.agent.retriever.holdout_ids)

    def test_predict_batch(self):
        batch = [
            {"thread_id": "test_1", "customer_text": "I was charged twice on my credit card"},
            {"thread_id": "test_2", "customer_text": "Wifi keeps disconnecting constantly"},
        ]
        results = self.agent.predict(batch)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["predicted_intent"], "billing_purchases_subscriptions")
        self.assertTrue(results[0]["predicted_escalate"])
        self.assertEqual(results[0]["predicted_escalate_reason"], "financial_billing_transaction")
        self.assertEqual(results[1]["predicted_intent"], "connectivity_network_issue")
        self.assertFalse(results[1]["predicted_escalate"])


if __name__ == "__main__":
    unittest.main()
