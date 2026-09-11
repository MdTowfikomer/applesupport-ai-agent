"""
Response Drafting Module (Task T8).
Drafts concise, grounded, empathetic Twitter replies as @AppleSupport.
Grounded in top-k historical resolutions retrieved from real Twitter support dialogues.
Supports Gemini LLM generation with seamless offline template fallback.
"""

import os
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


DRAFTER_SYSTEM_PROMPT = """You are the official @AppleSupport agent on Twitter.
Your mission is to draft a polite, highly concise, and helpful public Twitter reply.

STYLE GUIDELINES (AppleSupport Tone & Voice):
1. Brevity: Keep the reply under 35 words (fits comfortably in 280 characters).
2. Grounding: Base your troubleshooting advice strictly on the provided historical AppleSupport resolutions.
3. Tone: Warm, empathetic, professional, and direct.
4. Formatting:
   - Do NOT include bracketed placeholders like [Customer Name] or [Device].
   - Do NOT invent fake customer handles. Start directly with the response (e.g. "We'd like to help..." or "Let's look into this...").
5. Routing & Escalation:
   - If ESCALATE is TRUE: Empathize with the issue and invite the user to DM using the official link: "Please send us a DM so we can help: https://t.co/GDrqU22YpT".
   - If ESCALATE is FALSE:
     * If the issue has clear troubleshooting steps (e.g. restart, reset network settings, check battery usage in Settings): provide them directly.
     * If the issue is vague: ask a clarifying question about their device model and iOS version.
"""


class ResponseDrafter:
    """
    RAG-grounded Response Drafter using Gemini LLM with offline fallback.
    """

    def __init__(self, use_llm: bool = True, model_name: str = "gemini-flash-latest"):
        self.use_llm = use_llm
        self.model_name = os.getenv("PIPELINE_MODEL", model_name)
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self.config = None
        self.legacy_model = None

        if self.use_llm and self.api_key:
            try:
                from google import genai
                from google.genai import types
                self.client = genai.Client(api_key=self.api_key)
                self.config = types.GenerateContentConfig(
                    system_instruction=DRAFTER_SYSTEM_PROMPT,
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

    def _format_offline_reply(
        self,
        customer_text: str,
        intent: str,
        escalate: bool,
        escalate_reason: Optional[str],
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Constructs high-quality offline grounded reply from historical resolutions."""
        if intent == "vague_complaint_unclear":
            return (
                "We'd like to help get this resolved. "
                "Which device model are you using, and what iOS version is installed?"
            )

        if escalate:
            if escalate_reason in ["account_security_credentials", "financial_billing_transaction"]:
                return (
                    "We want to help ensure your account stays secure. "
                    "Please join us in a DM so we can look into this privately: https://t.co/GDrqU22YpT"
                )
            if escalate_reason == "physical_hardware_safety":
                return (
                    "We'd like to look into your hardware service options with you. "
                    "Please DM us your device details so we can assist: https://t.co/GDrqU22YpT"
                )
            return (
                "We've received your request and would like to look into this with you. "
                "Please connect with us in DM: https://t.co/GDrqU22YpT"
            )

        # If auto-handle and we have a retrieved resolution:
        if retrieved_context and len(retrieved_context) > 0:
            top_brand = retrieved_context[0].get("brand_text", "").strip()
            # Clean historical anonymized handle @\d+
            clean_brand = re.sub(r"^@\d+\s*", "", top_brand).strip()
            if clean_brand:
                return clean_brand

        # Default fallback
        return (
            "We're here to help. Have you tried restarting your device since this began? "
            "Let us know what version of iOS you're currently running."
        )

    def draft(
        self,
        customer_text: str,
        intent: str,
        escalate: bool,
        escalate_reason: Optional[str],
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Drafts a grounded Twitter customer support reply."""
        if not self.use_llm or (not self.client and not self.legacy_model):
            return self._format_offline_reply(
                customer_text, intent, escalate, escalate_reason, retrieved_context
            )

        try:
            context_block = ""
            if retrieved_context:
                examples = []
                for i, c in enumerate(retrieved_context[:3], 1):
                    examples.append(
                        f"Example {i} Customer: {c.get('customer_text')}\n"
                        f"Example {i} AppleSupport Reply: {c.get('brand_text')}"
                    )
                context_block = "\n\n".join(examples)

            prompt = (
                f"INCOMING CUSTOMER TWEET:\n\"{customer_text}\"\n\n"
                f"CLASSIFIED INTENT: {intent}\n"
                f"ESCALATE DECISION: {'TRUE (Human Escalation Required)' if escalate else 'FALSE (Auto-Handle)'}\n"
                f"ESCALATION REASON: {escalate_reason or 'None'}\n\n"
                f"SIMILAR HISTORICAL RESOLUTIONS FROM DATASET:\n{context_block}\n\n"
                f"Draft the Twitter reply for @AppleSupport:"
            )

            draft_text = ""

            # 1. Primary: Gemini
            if self.client:
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=self.config,
                    )
                    draft_text = response.text.strip()
                except Exception:
                    draft_text = ""

            # 2. Secondary: Groq fallback
            if not draft_text and self.groq_client:
                try:
                    g_resp = self.groq_client.chat.completions.create(
                        model="qwen/qwen3.8-27b",
                        messages=[
                            {"role": "system", "content": DRAFTER_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        max_tokens=100
                    )
                    draft_text = g_resp.choices[0].message.content.strip()
                except Exception:
                    draft_text = ""

            # Clean accidental markdown quotes
            draft_text = re.sub(r'^["\']|["\']$', "", draft_text).strip()
            # Clean accidental leading customer handle
            draft_text = re.sub(r"^@\w+\s*", "", draft_text).strip()

            if len(draft_text) > 0:
                return draft_text
            return self._format_offline_reply(
                customer_text, intent, escalate, escalate_reason, retrieved_context
            )
        except Exception:
            return self._format_offline_reply(
                customer_text, intent, escalate, escalate_reason, retrieved_context
            )
