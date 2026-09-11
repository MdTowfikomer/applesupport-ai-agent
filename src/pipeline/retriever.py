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
    In-memory Hybrid Retrieval Index (BM25 + Dense Semantic RRF) for historical customer support resolutions.
    Provides fast, top-k nearest-neighbor search over historical AppleSupport dialogues.
    """

    def __init__(
        self,
        subsample_path: str = "data/subsample/applesupport_threads_5k.jsonl",
        holdout_ids_path: str = "data/gold/index_holdout_ids.txt",
        use_semantic: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ):
        self.subsample_path = Path(subsample_path)
        self.holdout_ids_path = Path(holdout_ids_path)
        self.use_semantic = use_semantic
        self.embedding_model_name = embedding_model_name
        self.rrf_k = rrf_k
        self.embed_model = None
        self.corpus_embeddings = None
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
        self._init_semantic_model()

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

    def _init_semantic_model(self):
        """Initializes the semantic dense encoder and indexes the corpus if available."""
        if not self.use_semantic or not self.corpus:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self.embed_model = SentenceTransformer(self.embedding_model_name)
            corpus_texts = [d.get("customer_text", "") for d in self.corpus]
            self.corpus_embeddings = self.embed_model.encode(
                corpus_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
        except Exception:
            self.embed_model = None
            self.corpus_embeddings = None

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
                    "rrf_score": 0.0,
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
                    "rrf_score": 0.0,
                }
            ]

        # Precompute normalized BM25 scores for all matching candidates
        bm25_similarities: Dict[int, float] = {
            doc_idx: round(score / (self.doc_norms[doc_idx] * q_norm), 4)
            for doc_idx, score in scores.items()
        }

        bm25_ranked = sorted(
            scores.items(),
            key=lambda x: bm25_similarities[x[0]],
            reverse=True,
        )

        # Reciprocal Rank Fusion (RRF, k=60)
        rrf_scores: Dict[int, float] = defaultdict(float)
        for rank, (doc_idx, _) in enumerate(bm25_ranked):
            rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank + 1)

        # If dense semantic retriever is active, fuse dense ranks
        semantic_active = self.embed_model is not None and self.corpus_embeddings is not None
        if semantic_active:
            try:
                import numpy as np
                q_emb = self.embed_model.encode(
                    query, convert_to_numpy=True, normalize_embeddings=True
                )
                dense_sims = np.dot(self.corpus_embeddings, q_emb)
                dense_ranked = np.argsort(-dense_sims)
                for rank, doc_idx in enumerate(dense_ranked):
                    rrf_scores[int(doc_idx)] += 1.0 / (self.rrf_k + rank + 1)
            except Exception:
                semantic_active = False

        # Determine final candidate ordering
        if semantic_active:
            ranked_indices = [
                (doc_idx, rrf_scores[doc_idx])
                for doc_idx, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            ]
        else:
            ranked_indices = [
                (doc_idx, rrf_scores[doc_idx])
                for doc_idx, _ in bm25_ranked
            ]

        results = []
        for doc_idx, rrf_val in ranked_indices[:top_k]:
            doc = self.corpus[doc_idx]
            results.append(
                {
                    "thread_id": doc.get("thread_id"),
                    "customer_text": doc.get("customer_text"),
                    "brand_text": doc.get("brand_text"),
                    "score": bm25_similarities.get(doc_idx, 0.0),
                    "rrf_score": round(rrf_val, 6),
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
