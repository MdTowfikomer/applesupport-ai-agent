"""
AppleSupport AI Agent (Task T8).
Orchestrates the 4-stage pipeline:
  intent -> retrieve k -> draft -> escalate with reason

Components:
1. Intent: Canonical 10-class classifier adhering to Codebook rules.
2. Retrieve k: BM25/TF-IDF historical resolution retrieval over 4,800 non-holdout threads.
3. Draft: Context-grounded response generation via Gemini-2.5-flash with offline fallback.
4. Escalate: Enterprise policy and safety guardrail triage with explicit reason codes.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from src.pipeline.retriever import HistoricalResolutionIndex
from src.pipeline.classifier import IntentClassifier
from src.pipeline.triage import EscalationRouter
from src.pipeline.drafter import ResponseDrafter


class AppleSupportAgent:
    """
    Autonomous AI Customer Support Agent for @AppleSupport.
    """

    def __init__(
        self,
        subsample_path: str = "data/subsample/applesupport_threads_5k.jsonl",
        holdout_ids_path: str = "data/gold/index_holdout_ids.txt",
        use_llm: bool = True,
        top_k: int = 3,
        model_name: str = "gemini-2.5-flash",
    ):
        self.use_llm = use_llm
        self.top_k = top_k
        self.model_name = model_name

        # 1. Initialize Retrieval Index (strictly non-holdout)
        self.retriever = HistoricalResolutionIndex(
            subsample_path=subsample_path,
            holdout_ids_path=holdout_ids_path,
        )

        # 2. Initialize Pipeline Modules
        self.classifier = IntentClassifier(use_llm=use_llm, model_name=model_name)
        self.router = EscalationRouter()
        self.drafter = ResponseDrafter(use_llm=use_llm, model_name=model_name)

    def process(self, customer_text: str, thread_id: str = "") -> Dict[str, Any]:
        """
        Executes the full pipeline on a single incoming customer tweet:
        1. Classify Intent
        2. Retrieve Top-k Historical Resolutions
        3. Triage Escalation with Stated Reason
        4. Draft Grounded Reply
        """
        # Step 1: Intent Classification
        intent = self.classifier.classify(customer_text)

        # Step 2: Retrieve k historical candidate resolutions
        candidates = self.retriever.retrieve(customer_text, top_k=self.top_k)

        # Step 3: Escalation Triage with explicit reason
        escalate, escalate_reason = self.router.triage(
            customer_text=customer_text,
            predicted_intent=intent,
            retrieved_context=candidates,
        )

        # Step 4: Draft reply grounded in retrieved context
        reply = self.drafter.draft(
            customer_text=customer_text,
            intent=intent,
            escalate=escalate,
            escalate_reason=escalate_reason,
            retrieved_context=candidates,
        )

        top_match_id = candidates[0].get("thread_id") if candidates else None
        top_score = candidates[0].get("score", 0.0) if candidates else 0.0

        return {
            "thread_id": thread_id,
            "customer_text": customer_text,
            "predicted_intent": intent,
            "predicted_escalate": escalate,
            "predicted_escalate_reason": escalate_reason,
            "predicted_reply": reply,
            "retrieved_thread_id": top_match_id,
            "retrieval_score": top_score,
            "retrieved_candidates": candidates,
            "pipeline_model": self.model_name if self.use_llm else "offline_rule_tfidf",
        }

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Runs pipeline on a dictionary record from gold set or API."""
        return self.process(
            customer_text=record.get("customer_text", ""),
            thread_id=record.get("thread_id", ""),
        )

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a batch of customer records."""
        return [self.predict_record(r) for r in records]
