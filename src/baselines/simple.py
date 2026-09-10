"""
Simple Baseline Agent (Task T7).
Components:
1. Intent Classification: Keyword / regex rules derived from the canonical codebook
   (docs/intent_codebook.md), respecting the 5 codebook labeling rules & priority order.
2. Escalation Triage: Rule-based deterministic triggers for safety, legal, credentials,
   billing, channel transitions, and screenshot-only inputs.
3. Reply Drafting: Pure-Python TF-IDF retrieval over 4,800 historical AppleSupport
   threads, strictly isolating the 200 holdout records (zero data leakage).
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


INTENTS_ORDERED = [
    "account_access_security",
    "billing_purchases_subscriptions",
    "hardware_physical_accessory",
    "battery_power_issue",
    "connectivity_network_issue",
    "software_update_glitch",
    "performance_crash_freeze",
    "apps_feature_howto",
    "other",
    "vague_complaint_unclear",
]

NON_ENGLISH_PATTERN = re.compile(
    r"\b(voici|distance|oiseaux|chez|vivonne|ha\s+ancora|moltissimo|problemi|barra|"
    r"delle|lutfen|sorun|por\s*favor|buenos|dias|ayuda|gracias|merci|beaucoup|"
    r"que\s*hago|para|como|esta|hola|necesito|saludos)\b",
    re.IGNORECASE,
)

INTENT_KEYWORD_PATTERNS = {
    "account_access_security": re.compile(
        r"\b(apple\s*id|appleid|icloud|locked|lock\s*out|password|passcode|2fa|"
        r"two-?factor|verification\s*code|iforgot|security\s*questions?|account\s*recovery|"
        r"compromised|hacked|phishing|unauthorized\s*access|sign\s*in|log\s*in|login|"
        r"trusted\s*phone|trusted\s*number|disabled\s*in\s*the\s*itunes|fraudulent\s*email|"
        r"fake\s*email|scam\s*email)\b",
        re.IGNORECASE,
    ),
    "billing_purchases_subscriptions": re.compile(
        r"\b(charge[ds]?\s+(me|my|for|twice|again|to|on)|unauthorized\s*charge|"
        r"double\s*charge|overcharge|refund|subscription|subscriptions|purchase|purchased|"
        r"purchases|receipt|itunes\s*store|itunes\s*billing|bill|billing|cancel\s*subscription|"
        r"applecare|apple\s*care|payment\s*method|money\s*back|in-app\s*purchase|"
        r"reportaproblem|credit\s*card|debit\s*card)\b",
        re.IGNORECASE,
    ),
    "hardware_physical_accessory": re.compile(
        r"\b(screen|cracked|glass|broken|display|speaker|mic|microphone|headphone|"
        r"airpod|airpods|apple\s*watch|crown|home\s*button|genius\s*bar|repair|liquid|"
        r"water\s*damage|shattered|swollen|physical|hardware|jack|lightning\s*port|"
        r"earpiece|camera\s*lens)\b",
        re.IGNORECASE,
    ),
    "battery_power_issue": re.compile(
        r"\b(battery|drain|draining|drained|dying|percentage|charger|charging|charge|"
        r"overheat|overheating|getting\s*hot|so\s*hot|hot\s*phone|shut\s*off|dies\s*at|"
        r"battery\s*life|battery\s*health|dead\s*battery|power\s*down)\b",
        re.IGNORECASE,
    ),
    "connectivity_network_issue": re.compile(
        r"\b(wifi|wi-fi|bluetooth|cellular|network|no\s*sim|sim\s*card|captive|lte|"
        r"4g|3g|connection|disconnect|signal|carrier|hotspot|re-pairing|pairing|pair|"
        r"bluetooth\s*won)\b",
        re.IGNORECASE,
    ),
    "software_update_glitch": re.compile(
        r"\b(update|updated|updating|upgrade|upgraded|ios\s*11|ios11|11\.1|11\.0|11\.2|"
        r"patch|beta|autocorrect|keyboard|capital\s*i|letter\s*i|question\s*mark|"
        r"i[\uFE0E\uFE0F]?[\s]?[?]|glitch|glitches)\b",
        re.IGNORECASE,
    ),
    "performance_crash_freeze": re.compile(
        r"\b(freeze|freezing|frozen|lag|lagging|slow|unresponsive|crash|crashing|"
        r"crashes|crashed|restart|restarting|reboot|stutter|stuck|spinning\s*wheel|"
        r"black\s*screen|glitchy|spazzing|hang|hanging)\b",
        re.IGNORECASE,
    ),
    "apps_feature_howto": re.compile(
        r"\b(how\s*to|how\s*do\s*i|how\s*can\s*i|settings|apple\s*mail|mail\s*app|"
        r"apple\s*music|imessage|notes|photos|camera|workout|podcast|podcasts|"
        r"app\s*store|toggle|notification|notifications|alarm|library|playlist|"
        r"itunes\s*app|safari|siri|archive|delete\s*emails?|music\s*app|wrist\s*detection|"
        r"watch\s*app)\b",
        re.IGNORECASE,
    ),
    "other": re.compile(
        r"\b(store|where\s*can\s*i\s*find|address|buenos\s*aires|thanks|thank\s*you|"
        r"thx|fixed|resolved|solved|buy|buying|purchase\s*an\s*iphone|store\s*location)\b",
        re.IGNORECASE,
    ),
    "vague_complaint_unclear": re.compile(
        r"\b(broken|fix\s*this|worst|hate|sucks|suck|piece\s*of\s*shit|annoying|"
        r"terrible|useless|not\s*working|fix\s*your)\b",
        re.IGNORECASE,
    ),
}

ESCALATE_PATTERNS = [
    (
        "channel_transition",
        re.compile(
            r"\b(already\s+(in|sent|dm|dmed)|sent\s+(you\s+)?(a\s+)?dm|check\s+(your\s+)?dm|"
            r"in\s+(your\s+)?dm|dmd\s+you|dm\'d\s+you|pm\'d\s+you|check\s+pm|sent\s+dm)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "legal_regulatory_dispute",
        re.compile(
            r"\b(lawsuit|lawyer|sue|suing|attorney|legal\s*action|court|consumer\s*rights|"
            r"gdpr|police|fraud\s*report)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "physical_hardware_safety",
        re.compile(
            r"\b(swollen|swelling|smoke|smoking|spark|sparking|fire|burn|burning|burnt|"
            r"hot\s*to\s*touch|explode|exploding)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "account_security_credentials",
        re.compile(
            r"\b(apple\s*id\s+locked|locked\s+apple\s*id|account\s+locked|two-?factor|2fa|"
            r"verification\s*code|security\s*questions?|iforgot|reset\s*password|"
            r"compromised\s*account|hacked\s*account)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "financial_billing_transaction",
        re.compile(
            r"\b(unauthorized\s*charge|charged\s*twice|double\s*charge|refund|"
            r"cancel\s*(my\s*)?subscription|chargeback|fraudulent\s*charge)\b",
            re.IGNORECASE,
        ),
    ),
]

SCREENSHOT_ONLY_PATTERN = re.compile(r"^(@\w+\s*)*(https?://\S+)?\s*$", re.IGNORECASE)


def tokenize_words(text: str) -> List[str]:
    """Tokenizes alphanumeric text tokens in lower case."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class TfidfRetriever:
    """
    Pure-Python TF-IDF inverted index retriever.
    Operates offline without external C-extensions or network calls.
    """

    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.df: Counter = Counter()
        self.doc_tokens: List[List[str]] = []

        for doc in corpus:
            tokens = tokenize_words(doc.get("customer_text", ""))
            self.df.update(set(tokens))
            self.doc_tokens.append(tokens)

        # Smooth IDF: log((N + 1) / (df + 1)) + 1
        self.idf: Dict[str, float] = {
            t: math.log((self.doc_count + 1) / (df_val + 1)) + 1.0
            for t, df_val in self.df.items()
        }

        # Precompute inverted index and document L2 norms
        self.inverted_index: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.doc_norms: List[float] = []

        for i, tokens in enumerate(self.doc_tokens):
            tf = Counter(tokens)
            norm_sq = 0.0
            for t, count in tf.items():
                weight = count * self.idf[t]
                norm_sq += weight * weight
                self.inverted_index[t].append((i, weight))
            self.doc_norms.append(math.sqrt(norm_sq) if norm_sq > 0 else 1.0)

    def search(self, query: str, top_k: int = 1) -> List[Tuple[int, float]]:
        """Finds top-k documents in corpus by cosine similarity."""
        q_tokens = tokenize_words(query)
        if not q_tokens:
            return [(0, 0.0)]

        q_tf = Counter(q_tokens)
        q_weights = {
            t: count * self.idf.get(t, 0.0)
            for t, count in q_tf.items()
            if t in self.idf
        }
        q_norm_sq = sum(w * w for w in q_weights.values())
        if q_norm_sq <= 0:
            return [(0, 0.0)]

        q_norm = math.sqrt(q_norm_sq)
        scores: Dict[int, float] = defaultdict(float)

        for t, q_w in q_weights.items():
            for doc_idx, doc_w in self.inverted_index[t]:
                scores[doc_idx] += q_w * doc_w

        if not scores:
            return [(0, 0.0)]

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1] / (self.doc_norms[x[0]] * q_norm),
            reverse=True,
        )
        return [
            (idx, score / (self.doc_norms[idx] * q_norm))
            for idx, score in ranked[:top_k]
        ]


class SimpleBaselineAgent:
    """
    Simple Baseline Agent (T7):
    - Intent: Keyword rule pattern classifier respecting codebook hierarchy.
    - Escalation: Rule-based triage with explicit reason codes.
    - Reply: TF-IDF nearest-neighbor retrieval over non-holdout historical threads.
    """

    def __init__(
        self,
        subsample_path: Optional[str] = "data/subsample/applesupport_threads_5k.jsonl",
        holdout_ids_path: Optional[str] = "data/gold/index_holdout_ids.txt",
    ):
        self.subsample_path = subsample_path
        self.holdout_ids_path = holdout_ids_path
        self.holdout_ids: Set[str] = set()
        self.corpus: List[Dict[str, Any]] = []
        self.retriever: Optional[TfidfRetriever] = None

        if subsample_path and holdout_ids_path:
            self._initialize_from_paths(subsample_path, holdout_ids_path)

    def _initialize_from_paths(self, subsample_path: str, holdout_ids_path: str):
        sub_p = Path(subsample_path)
        hold_p = Path(holdout_ids_path)

        if hold_p.exists():
            with open(hold_p, "r", encoding="utf-8") as f:
                self.holdout_ids = {line.strip() for line in f if line.strip()}

        if sub_p.exists():
            clean_corpus = []
            with open(sub_p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    # Strict isolation assertion: NO holdout ID in retrieval corpus
                    if item.get("thread_id") not in self.holdout_ids:
                        clean_corpus.append(item)
            self.corpus = clean_corpus
            self.retriever = TfidfRetriever(self.corpus)

    def fit(self, corpus: List[Dict[str, Any]], holdout_ids: Optional[Set[str]] = None):
        """Fits agent on an explicit corpus and holdout set."""
        self.holdout_ids = holdout_ids or set()
        self.corpus = [
            doc for doc in corpus if doc.get("thread_id") not in self.holdout_ids
        ]
        self.retriever = TfidfRetriever(self.corpus)
        return self

    def predict_intent(self, text: str) -> str:
        """
        Classifies intent using codebook regex profiles and priority rules.
        """
        clean_text = text.lower()

        # 1. Non-English check -> other
        if NON_ENGLISH_PATTERN.search(clean_text):
            return "other"

        # 2. Score matches across intents
        matches = {
            intent: len(INTENT_KEYWORD_PATTERNS[intent].findall(clean_text))
            for intent in INTENTS_ORDERED
            if INTENT_KEYWORD_PATTERNS[intent].search(clean_text)
        }

        # 3. Apply strict codebook priority order
        for intent in INTENTS_ORDERED:
            if intent in matches:
                return intent

        # 4. Fallback for unclassified text: questions/complaints vs non-support
        if "?" in text or any(
            w in clean_text for w in ["why", "what", "how", "when", "help", "please"]
        ):
            return "vague_complaint_unclear"

        return "other"

    def predict_escalation(
        self, text: str, predicted_intent: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Applies deterministic escalation rules to customer text and predicted intent.
        """
        clean_text = text.lower()

        # Missing context / screenshot-only rule
        if SCREENSHOT_ONLY_PATTERN.match(text.strip()):
            return True, "missing_context_screenshot"

        # High priority emergency triggers
        for reason, pat in ESCALATE_PATTERNS:
            if pat.search(clean_text):
                return True, reason

        # Mandatory escalation intents as defined by Codebook Section 5.2
        if predicted_intent == "account_access_security":
            return True, "account_security_credentials"
        if predicted_intent == "billing_purchases_subscriptions":
            return True, "financial_billing_transaction"
        if predicted_intent == "hardware_physical_accessory":
            return True, "physical_hardware_safety"

        return False, None

    def retrieve_reply(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Drafts reply grounded in historical AppleSupport resolutions via TF-IDF.
        """
        if not self.retriever or not self.corpus:
            default_reply = (
                "Thanks for reaching out to us. We'd like to help get this resolved. "
                "Please send us a DM so we can look into this with you: https://t.co/GDrqU22YpT"
            )
            return default_reply, {"retrieved_thread_id": None, "retrieval_score": 0.0}

        top_idx, score = self.retriever.search(text, top_k=1)[0]
        matched_doc = self.corpus[top_idx]
        reply_text = matched_doc.get("brand_text", "")
        metadata = {
            "retrieved_thread_id": matched_doc.get("thread_id"),
            "retrieval_score": round(score, 4),
        }
        return reply_text, metadata

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Runs end-to-end inference on a single customer tweet record."""
        text = record.get("customer_text", "")
        intent = self.predict_intent(text)
        escalate, escalate_reason = self.predict_escalation(text, intent)
        reply, meta = self.retrieve_reply(text)

        return {
            "thread_id": record.get("thread_id"),
            "customer_tweet_id": record.get("customer_tweet_id"),
            "customer_text": text,
            "predicted_intent": intent,
            "predicted_escalate": escalate,
            "predicted_escalate_reason": escalate_reason,
            "predicted_reply": reply,
            "retrieved_thread_id": meta.get("retrieved_thread_id"),
            "retrieval_score": meta.get("retrieval_score"),
            "baseline_name": "simple_baseline_agent",
        }

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Runs batch prediction over an iterable of records."""
        return [self.predict_record(r) for r in records]
