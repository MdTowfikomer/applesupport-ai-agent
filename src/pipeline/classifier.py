"""
Intent Classifier Module (Task T8).
Classifies customer inbound tweets into the canonical 10-class AppleSupport taxonomy:
1. battery_power_issue
2. performance_crash_freeze
3. software_update_glitch
4. connectivity_network_issue
5. hardware_physical_accessory
6. account_access_security
7. billing_purchases_subscriptions
8. apps_feature_howto
9. vague_complaint_unclear
10. other

Adheres strictly to the 5 ground-truth labeling rules from docs/intent_codebook.md.
Supports Gemini LLM structured classification with seamless offline regex fallback.
"""

import os
import json
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from src.evaluation.validator import ALLOWED_INTENTS
from src.baselines.simple import (
    INTENT_KEYWORD_PATTERNS,
    INTENTS_ORDERED,
    NON_ENGLISH_PATTERN,
)


INTENT_PROMPT_SYSTEM = """You are an expert customer support intent classifier for @AppleSupport on Twitter.
Your job is to classify the customer's first inbound tweet into EXACTLY ONE of these 10 canonical intents:

1. battery_power_issue: Rapid battery drain, unexpected shutoffs at non-zero %, overheating while charging/use, battery health.
2. performance_crash_freeze: Lag, unresponsiveness, freezing on home screen/apps, third-party or system app crashes, spinning wheels.
3. software_update_glitch: OS defect/bug after updating (e.g. autocorrect 'I' to 'A [?]', blank notification curtain, patch regressions).
4. connectivity_network_issue: Wi-Fi dropping, Bluetooth accessories failing to connect, cellular data loss, "No SIM" alert.
5. hardware_physical_accessory: Physical damage (cracked screen/back glass), liquid spill, broken speaker/mic, AirPods physical defect, Genius Bar booking.
6. account_access_security: Apple ID locked, 2FA/verification codes, password reset, security questions, phishing/suspicious email alert.
7. billing_purchases_subscriptions: Unauthorized credit card charges, refund requests, accidental in-app purchases, canceling recurring subscriptions.
8. apps_feature_howto: How to use built-in features (Apple Mail, Music, iMessage, Watch Workout, Podcasts, Camera, Photos settings).
9. vague_complaint_unclear: Generic complaint or anger with ZERO symptoms and NO OS update named (e.g. "my phone is broken", "fix this").
10. other: Non-English inquiries, store locator requests, buying advice, or conversational closures ("thanks!", "all good").

CRITICAL LABELING RULES:
- Rule 1 (Inbound Primacy): Base your decision ONLY on the customer's text.
- Rule 2 (Channel Transition): A mention of DM does NOT make the intent "other"; classify by the stated issue.
- Rule 3 (Precedence): account_access_security > billing_purchases_subscriptions > hardware_physical_accessory > symptoms > vague_complaint_unclear.
- Rule 4 (Symptom/Update Rule): If the customer names ANY symptom or ANY OS version/update, DO NOT use vague_complaint_unclear.
- Output strictly JSON: {"intent": "<exact_intent_code>", "reasoning": "<brief explanation>"}
"""


class IntentClassifier:
    """
    Hybrid Intent Classifier supporting Gemini LLM structured generation
    with deterministic codebook rule fallback.
    """

    def __init__(self, use_llm: bool = True, model_name: str = "gemini-flash-latest"):
        self.use_llm = use_llm
        self.model_name = os.getenv("PIPELINE_MODEL", model_name)
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self.config = None
        if self.use_llm and self.api_key:
            try:
                from google import genai
                from google.genai import types
                self.client = genai.Client(api_key=self.api_key)
                self.config = types.GenerateContentConfig(
                    system_instruction=INTENT_PROMPT_SYSTEM,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
            except Exception:
                self.client = None

        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = None
        if self.use_llm and self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception:
                self.groq_client = None

    def classify_rule_based(self, text: str) -> str:
        """Deterministic keyword pattern matching complying with Codebook rules."""
        clean_text = text.lower()

        # Non-English check
        if NON_ENGLISH_PATTERN.search(clean_text):
            return "other"

        # Match across intents
        matches = {
            intent: len(INTENT_KEYWORD_PATTERNS[intent].findall(clean_text))
            for intent in INTENTS_ORDERED
            if INTENT_KEYWORD_PATTERNS[intent].search(clean_text)
        }

        # Apply codebook priority order
        for intent in INTENTS_ORDERED:
            if intent in matches:
                return intent

        # Fallback for unclassified queries
        if "?" in text or any(
            w in clean_text for w in ["why", "what", "how", "when", "help", "please"]
        ):
            return "vague_complaint_unclear"

        return "other"

    def classify(self, text: str) -> str:
        """Classifies customer message into one of 10 canonical intents."""
        if not self.use_llm or (not self.client and not self.groq_client):
            return self.classify_rule_based(text)

        prompt = f'Customer Tweet: "{text}"\nClassify the intent JSON:'
        raw_text = ""

        # 1. Primary: Gemini
        if self.client:
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.config,
                )
                raw_text = resp.text.strip()
            except Exception:
                raw_text = ""

        # 2. Secondary: Groq (ultra-fast fallback on quota limit)
        if not raw_text and self.groq_client:
            try:
                g_resp = self.groq_client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=[
                        {"role": "system", "content": INTENT_PROMPT_SYSTEM},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                raw_text = g_resp.choices[0].message.content.strip()
            except Exception:
                raw_text = ""

        if not raw_text:
            return self.classify_rule_based(text)

        try:
            # Strip markdown fence if present
            clean_raw = re.sub(r"^```json\s*", "", raw_text)
            clean_raw = re.sub(r"\s*```$", "", clean_raw).strip()
            data = json.loads(clean_raw)
            pred_intent = data.get("intent", "").strip()

            if pred_intent in ALLOWED_INTENTS:
                return pred_intent
        except Exception:
            pass

        return self.classify_rule_based(text)