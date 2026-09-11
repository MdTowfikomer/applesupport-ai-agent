"""
Escalation Triage Router (Task T8).
Decides whether an incoming customer inquiry should be auto-handled or escalated to a human agent,
along with a structured, policy-compliant reason code.

Canonical Reason Codes:
1. account_security_credentials
2. financial_billing_transaction
3. physical_hardware_safety
4. legal_regulatory_dispute
5. channel_transition
6. missing_context_screenshot
"""

import re
from typing import Tuple, Optional, Dict, Any, List

from src.baselines.simple import (
    ESCALATE_PATTERNS,
    SCREENSHOT_ONLY_PATTERN,
)


FRUSTRATION_INTENTS = {
    "performance_crash_freeze",
    "software_update_glitch",
    "connectivity_network_issue",
}

PROFANITY_PATTERN = re.compile(
    r"\b(fuck|fucking|fucked|shit|bullshit|damn|crap|ass|piss|pissed|hell)\b",
    re.IGNORECASE,
)
EXCESSIVE_PUNCTUATION_PATTERN = re.compile(r"[!?]{3,}")
ALL_CAPS_WORD_PATTERN = re.compile(r"\b[A-Z]{4,}\b")


class EscalationRouter:
    """
    Enterprise Escalation Router implementing deterministic safety guardrails
    and intent-driven triage policies derived from docs/intent_codebook.md.
    """

    def __init__(self, trust_threshold: float = 0.85):
        self.trust_threshold = trust_threshold

    def triage(
        self,
        customer_text: str,
        predicted_intent: str,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determines binary escalation decision and reason code.
        Returns: (escalate: bool, escalate_reason: Optional[str])
        """
        clean_text = customer_text.lower().strip()

        # 1. Missing context / Screenshot-only rule
        if SCREENSHOT_ONLY_PATTERN.match(customer_text.strip()):
            return True, "missing_context_screenshot"

        # 2. High-priority emergency / policy triggers
        for reason, pat in ESCALATE_PATTERNS:
            if pat.search(clean_text):
                return True, reason

        # 3. Codebook Section 5.2 Mandatory Escalation Intents
        if predicted_intent == "account_access_security":
            return True, "account_security_credentials"
        if predicted_intent == "billing_purchases_subscriptions":
            return True, "financial_billing_transaction"
        if predicted_intent == "hardware_physical_accessory":
            return True, "physical_hardware_safety"

        # 4. Active DM transition signal from historical resolutions
        # If the top retrieved resolution indicates that AppleSupport always escalates this specific scenario to DM
        if retrieved_context and len(retrieved_context) > 0:
            top_match = retrieved_context[0]
            top_brand = top_match.get("brand_text", "").lower()
            top_score = top_match.get("score", 0.0)

            # If strong query match (>0.60) and historical brand response was purely an intake DM transition:
            if top_score >= 0.60 and any(
                dm_phrase in top_brand
                for dm_phrase in [
                    "received your dm",
                    "got your dm",
                    "meet us in dm",
                    "join us in a dm",
                ]
            ):
                return True, "channel_transition"

        # 5. Secondary channel_transition signal for high-frustration technical complaints
        if predicted_intent in FRUSTRATION_INTENTS and retrieved_context and len(retrieved_context) > 0:
            top_score = retrieved_context[0].get("score", 0.0)
            if top_score > 0.15:
                has_profanity = bool(PROFANITY_PATTERN.search(customer_text))
                has_excessive_punct = bool(EXCESSIVE_PUNCTUATION_PATTERN.search(customer_text))
                has_all_caps = bool(ALL_CAPS_WORD_PATTERN.search(customer_text))

                if has_profanity or has_excessive_punct or has_all_caps:
                    return True, "channel_transition"

        # 6. Default auto-handle: deterministic technical troubleshooting or clarifying intake
        return False, None
