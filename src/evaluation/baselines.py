"""
Dummy Sanity-Check Baselines (T5).

DISCLAIMER:
These are non-learning, rule-free sanity-check baselines designed to establish
the empirical lower-bound benchmark performance on data/gold/gold_eval_200.jsonl.
They DO NOT use machine learning, retrieval, embeddings, LLMs, or trained models.
"""

from collections import Counter
from typing import List, Dict, Any, Optional

class MajorityIntentBaseline:
    """
    Majority-Intent Sanity Check Baseline.
    
    Identifies the single most frequent intent in the dataset and always predicts that class.
    Used to measure the statistical floor of multiclass intent classification.
    """
    def __init__(self, majority_intent: Optional[str] = None):
        self.majority_intent = majority_intent

    def fit(self, gold_records: List[Dict[str, Any]]) -> "MajorityIntentBaseline":
        """Calculates majority class from provided gold records."""
        intents = [r["intent"] for r in gold_records if r.get("intent")]
        if not intents:
            raise ValueError("No valid intents found to determine majority class")
        counts = Counter(intents)
        self.majority_intent = counts.most_common(1)[0][0]
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates predictions for each record.
        Outputs in-memory prediction dictionaries without altering input/gold data.
        """
        if self.majority_intent is None:
            raise ValueError("Baseline has not been fit or provided with a majority intent.")
        
        predictions = []
        for r in records:
            predictions.append({
                "thread_id": r["thread_id"],
                "predicted_intent": self.majority_intent,
                # For baseline purposes, defaults escalate to False or keeps separate
                "predicted_escalate": False,
                "baseline_name": "majority_intent"
            })
        return predictions


class AlwaysEscalateBaseline:
    """
    Always-Escalate Sanity Check Baseline.
    
    Always predicts `escalate = True` regardless of customer inquiry or technical context.
    Demonstrates the trade-off of 100% human routing (high recall, zero automation precision).
    """
    def __init__(self, default_intent: str = "other"):
        self.default_intent = default_intent

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates predictions for each record.
        Outputs in-memory prediction dictionaries without altering input/gold data.
        """
        predictions = []
        for r in records:
            predictions.append({
                "thread_id": r["thread_id"],
                "predicted_intent": self.default_intent,
                "predicted_escalate": True,
                "baseline_name": "always_escalate"
            })
        return predictions


class TrivialBaselineAgent:
    """
    Trivial End-to-End Baseline Agent (Task T6).
    
    A complete, non-learning baseline support agent that implements all 3 required capabilities:
    1. Intent Classification: Predicts majority intent from training/gold distribution ('other').
    2. Escalation Triage: Predicts majority escalation decision (escalate=True, reason='channel_transition').
    3. Reply Drafting: Emits AppleSupport's historical most frequent canned template:
       "Thanks for reaching out to us. We'd like to help get this resolved. Please send us a DM so we can look into this with you: https://t.co/GDrqU22YpT"
       
    This baseline serves as the empirical performance benchmark for measuring the value added
    by subsequent retrieval-augmented generation (RAG) and calibrated decision pipelines.
    """
    DEFAULT_CANNED_REPLY = (
        "Thanks for reaching out to us. We'd like to help get this resolved. "
        "Please send us a DM so we can look into this with you: https://t.co/GDrqU22YpT"
    )

    def __init__(
        self,
        majority_intent: str = "other",
        escalate: bool = True,
        escalate_reason: Optional[str] = "channel_transition",
        canned_reply: Optional[str] = None
    ):
        self.majority_intent = majority_intent
        self.escalate = escalate
        self.escalate_reason = escalate_reason
        self.canned_reply = canned_reply or self.DEFAULT_CANNED_REPLY

    def fit(self, gold_records: List[Dict[str, Any]]) -> "TrivialBaselineAgent":
        """Computes majority intent and majority escalation decision from records."""
        intents = [r["intent"] for r in gold_records if r.get("intent")]
        if intents:
            self.majority_intent = Counter(intents).most_common(1)[0][0]
            
        escalates = [r["escalate"] for r in gold_records if r.get("escalate") is not None]
        if escalates:
            self.escalate = Counter(escalates).most_common(1)[0][0]
            if self.escalate:
                reasons = [r["escalate_reason"] for r in gold_records if r.get("escalate_reason")]
                if reasons:
                    self.escalate_reason = Counter(reasons).most_common(1)[0][0]
            else:
                self.escalate_reason = None
        return self

    def predict_one(self, customer_text: str, thread_id: str = "") -> Dict[str, Any]:
        """Generates full prediction for a single incoming customer message."""
        return {
            "thread_id": thread_id,
            "predicted_intent": self.majority_intent,
            "predicted_escalate": self.escalate,
            "predicted_escalate_reason": self.escalate_reason,
            "predicted_reply": self.canned_reply,
            "baseline_name": "trivial_baseline_agent"
        }

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates predictions for a batch of records without modifying input data.
        Returns a list of prediction dictionaries.
        """
        predictions = []
        for r in records:
            tid = r.get("thread_id", "")
            ct = r.get("customer_text", "")
            pred = self.predict_one(customer_text=ct, thread_id=tid)
            predictions.append(pred)
        return predictions

