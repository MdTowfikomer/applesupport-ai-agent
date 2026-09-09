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
