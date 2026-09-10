"""
Historical Resolution Retrieval Index (Task T8).
Indexes 4,800 non-holdout AppleSupport threads from data/subsample/applesupport_threads_5k.jsonl.
Strictly excludes all 200 holdout IDs from data/gold/index_holdout_ids.txt.
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class HistoricalResolutionIndex:
    """
    In-memory BM25/TF-IDF Retrieval Index for historical customer support resolutions.
    Provides fast, deterministic top-k nearest-neighbor search over historical AppleSupport dialogues.
    """

    def __init__(
        self,
        subsample_path: str = "data/subsample/applesupport_threads_5k.jsonl",
        holdout_ids_path: str = "data/gold/index_holdout_ids.txt",
    ):
        self.subsample_path = Path(subsample_path)
        self.holdout_ids_path = Path(holdout_ids_path)
        self.holdout_ids: Set[str] = set()
        self.corpus: List[Dict[str, Any]] = []

        self.df: Counter = Counter()
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.doc_norms: List[float] = []

        self._load_and_index()

    def _load_and_index(self):
        """Loads non-holdout corpus and constructs the inverted index."""
        # 1. Load holdout IDs
        if self.holdout_ids_path.exists():
            with open(self.holdout_ids_path, "r", encoding="utf-8") as f:
                self.holdout_ids = {line.strip() for line in f if line.strip()}

        # 2. Load and filter historical records
        if self.subsample_path.exists():
            clean_docs = []
            with open(self.subsample_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    tid = item.get("thread_id")
                    # Strict isolation assertion: NO holdout ID allowed in retrieval index
                    if tid and tid not in self.holdout_ids:
                        clean_docs.append(item)
            self.corpus = clean_docs

        # 3. Assert zero leakage
        assert not any(
            d.get("thread_id") in self.holdout_ids for d in self.corpus
        ), "CRITICAL: Data leakage detected in retrieval index!"

        self.doc_count = len(self.corpus)
        if self.doc_count == 0:
            return

        # 4. Compute document frequencies and lengths
        total_len = 0
        for doc in self.corpus:
            tokens = tokenize(doc.get("customer_text", ""))
            self.df.update(set(tokens))
            self.doc_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len

        self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 1.0

        # 5. Compute smoothed IDF: log((N - df + 0.5) / (df + 0.5) + 1.0)
        for term, df_val in self.df.items():
            self.idf[term] = math.log((self.doc_count - df_val + 0.5) / (df_val + 0.5) + 1.0)

        # 6. Build inverted index with BM25 term weighting (k1=1.5, b=0.75)
        k1 = 1.5
        b = 0.75
        for idx, tokens in enumerate(self.doc_tokens):
            tf = Counter(tokens)
            doc_len = self.doc_lengths[idx]
            len_norm = (1.0 - b) + b * (doc_len / self.avg_doc_len)
            norm_sq = 0.0

            for term, count in tf.items():
                bm25_tf = (count * (k1 + 1)) / (count + k1 * len_norm)
                weight = bm25_tf * self.idf[term]
                norm_sq += weight * weight
                self.inverted_index[term].append((idx, weight))

            self.doc_norms.append(math.sqrt(norm_sq) if norm_sq > 0 else 1.0)

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most relevant historical AppleSupport resolution threads.
        """
        if not self.corpus:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            # Fallback to first doc if query is empty
            first = self.corpus[0]
            return [
                {
                    "thread_id": first.get("thread_id"),
                    "customer_text": first.get("customer_text"),
                    "brand_text": first.get("brand_text"),
                    "score": 0.0,
                }
            ]

        q_tf = Counter(q_tokens)
        q_weights = {
            t: count * self.idf.get(t, 0.0)
            for t, count in q_tf.items()
            if t in self.idf
        }
        q_norm_sq = sum(w * w for w in q_weights.values())
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

        scores: Dict[int, float] = defaultdict(float)
        for term, q_w in q_weights.items():
            for doc_idx, doc_w in self.inverted_index[term]:
                scores[doc_idx] += q_w * doc_w

        if not scores:
            first = self.corpus[0]
            return [
                {
                    "thread_id": first.get("thread_id"),
                    "customer_text": first.get("customer_text"),
                    "brand_text": first.get("brand_text"),
                    "score": 0.0,
                }
            ]

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1] / (self.doc_norms[x[0]] * q_norm),
            reverse=True,
        )

        results = []
        for doc_idx, score in ranked[:top_k]:
            doc = self.corpus[doc_idx]
            sim = score / (self.doc_norms[doc_idx] * q_norm)
            results.append(
                {
                    "thread_id": doc.get("thread_id"),
                    "customer_text": doc.get("customer_text"),
                    "brand_text": doc.get("brand_text"),
                    "score": round(sim, 4),
                }
            )

        return results

    def format_context(self, candidates: List[Dict[str, Any]]) -> str:
        """Formats retrieved candidate threads into a structured prompt context block."""
        if not candidates:
            return "No historical resolutions found."

        blocks = []
        for i, c in enumerate(candidates, 1):
            blocks.append(
                f"[Example {i}] (Thread: {c.get('thread_id')}, Match Score: {c.get('score', 0.0)})\n"
                f"Customer Query: {c.get('customer_text')}\n"
                f"AppleSupport Resolution: {c.get('brand_text')}"
            )
        return "\n\n".join(blocks)
