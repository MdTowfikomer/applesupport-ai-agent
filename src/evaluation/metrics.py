"""
Evaluation Metrics Module for AppleSupport Benchmark.
Computes:
- Multiclass Intent Accuracy, Macro-F1, Per-Class Precision/Recall/F1/Support
- Binary Escalation Accuracy, Precision, Recall, F1, Confusion Matrix
- ASCII / Text table formatters for terminal reports and documentation
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

def compute_accuracy(y_true: List[Any], y_pred: List[Any]) -> float:
    """Computes exact-match accuracy."""
    if not y_true:
        return 0.0
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)

def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str]
) -> List[List[int]]:
    """
    Computes confusion matrix where rows are actual (true) and columns are predicted.
    cm[i][j] = count of items with actual label labels[i] and predicted label labels[j].
    """
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n = len(labels)
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            matrix[label_to_idx[yt]][label_to_idx[yp]] += 1

    return matrix

def compute_multiclass_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes accuracy, macro-F1, per-class precision/recall/F1/support,
    and confusion matrix for multiclass classification.
    """
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    
    if labels is None:
        labels = sorted(list(set(y_true).union(set(y_pred))))

    cm = compute_confusion_matrix(y_true, y_pred, labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    
    per_class = {}
    f1_scores = []
    
    for lbl in labels:
        idx = label_to_idx[lbl]
        tp = cm[idx][idx]
        fp = sum(cm[r][idx] for r in range(len(labels)) if r != idx)
        fn = sum(cm[idx][c] for c in range(len(labels)) if c != idx)
        support = sum(cm[idx][c] for c in range(len(labels)))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[lbl] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        f1_scores.append(f1)

    accuracy = compute_accuracy(y_true, y_pred)
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    # Weighted-F1 for reference
    total_support = sum(p["support"] for p in per_class.values())
    weighted_f1 = (
        sum(p["f1"] * p["support"] for p in per_class.values()) / total_support
        if total_support > 0 else 0.0
    )

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total_samples": len(y_true),
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm
    }

def compute_binary_escalation_metrics(
    y_true: List[bool],
    y_pred: List[bool]
) -> Dict[str, Any]:
    """
    Computes binary classification metrics for escalation where True is the positive class.
    Returns: accuracy, precision, recall, f1, TP, FP, TN, FN, and count table.
    """
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt is True and yp is True)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt is False and yp is True)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt is False and yp is False)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt is True and yp is False)
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    pos_support = tp + fn
    neg_support = tn + fp
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_support": pos_support,
        "negative_support": neg_support,
        "total_samples": total,
        "confusion_matrix": {
            "actual_true_pred_true (TP)": tp,
            "actual_false_pred_true (FP)": fp,
            "actual_false_pred_false (TN)": tn,
            "actual_true_pred_false (FN)": fn
        }
    }

def format_multiclass_report(metrics: Dict[str, Any]) -> str:
    """Formats multiclass evaluation results into a clean markdown table."""
    lines = []
    lines.append(f"**Overall Accuracy**: `{metrics['accuracy'] * 100:.2f}%` ({int(metrics['accuracy'] * metrics['total_samples'])}/{metrics['total_samples']})")
    lines.append(f"**Macro-F1**: `{metrics['macro_f1'] * 100:.2f}%` | **Weighted-F1**: `{metrics['weighted_f1'] * 100:.2f}%`\n")
    lines.append("| Intent Class | Precision | Recall | F1-Score | Support |")
    lines.append("|:---|:---:|:---:|:---:|:---:|")
    
    for lbl, p in metrics["per_class"].items():
        lines.append(f"| `{lbl}` | {p['precision'] * 100:.1f}% | {p['recall'] * 100:.1f}% | {p['f1'] * 100:.1f}% | {p['support']} |")
    
    return "\n".join(lines)

def format_confusion_matrix_ascii(cm: List[List[int]], labels: List[str], max_name_len: int = 24) -> str:
    """Formats confusion matrix into readable ASCII table."""
    # Truncate labels for display
    short_labels = [l if len(l) <= max_name_len else l[:max_name_len-2] + ".." for l in labels]
    indices = [f"[{i+1}]" for i in range(len(labels))]
    
    lines = []
    # Header legend
    lines.append("Class Key:")
    for idx_str, name in zip(indices, labels):
        lines.append(f"  {idx_str:>4} = {name}")
    lines.append("\nConfusion Matrix (Rows: Actual / Columns: Predicted):")
    
    # Column header
    col_hdr = f"{'Actual \\ Pred':<26}" + "".join(f"{idx_str:>6}" for idx_str in indices) + f"{'Total':>7}"
    lines.append(col_hdr)
    lines.append("-" * len(col_hdr))
    
    for i, (name, row) in enumerate(zip(short_labels, cm)):
        row_str = f"{indices[i]:>4} {name:<21}" + "".join(f"{v:>6}" for v in row) + f"{sum(row):>7}"
        lines.append(row_str)
        
    lines.append("-" * len(col_hdr))
    total_preds = [sum(cm[r][c] for r in range(len(labels))) for c in range(len(labels))]
    lines.append(f"{'Total Predicted':<26}" + "".join(f"{v:>6}" for v in total_preds) + f"{sum(total_preds):>7}")
    
    return "\n".join(lines)

def format_binary_report(metrics: Dict[str, Any]) -> str:
    """Formats binary escalation metrics into clean markdown."""
    lines = []
    lines.append(f"- **Escalation Accuracy**: `{metrics['accuracy'] * 100:.2f}%` ({metrics['tp'] + metrics['tn']}/{metrics['total_samples']})")
    lines.append(f"- **Precision (escalate=True)**: `{metrics['precision'] * 100:.2f}%` ({metrics['tp']}/{metrics['tp'] + metrics['fp']})")
    lines.append(f"- **Recall (escalate=True)**: `{metrics['recall'] * 100:.2f}%` ({metrics['tp']}/{metrics['tp'] + metrics['fn']})")
    lines.append(f"- **F1-Score (escalate=True)**: `{metrics['f1'] * 100:.2f}%`\n")
    lines.append("| Actual \\ Predicted | Pred: Escalate (True) | Pred: Auto-Handle (False) | Total Actual |")
    lines.append("|:---|:---:|:---:|:---:|")
    lines.append(f"| **Actual: Escalate (True)** | **TP = {metrics['tp']}** | **FN = {metrics['fn']}** | {metrics['positive_support']} |")
    lines.append(f"| **Actual: Auto-Handle (False)** | **FP = {metrics['fp']}** | **TN = {metrics['tn']}** | {metrics['negative_support']} |")
    lines.append(f"| **Total Predicted** | {metrics['tp'] + metrics['fp']} | {metrics['fn'] + metrics['tn']} | {metrics['total_samples']} |")
    return "\n".join(lines)


# ==============================================================================
# Text Reply Lexical Metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU-1, Length Stats)
# ==============================================================================

import re
import math

def tokenize_text(text: str) -> List[str]:
    """Lowercase alphanumeric word tokenization."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())

def compute_ngrams(tokens: List[str], n: int) -> Counter:
    """Extracts n-gram frequency counter."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

def compute_ngram_overlap(ref_tokens: List[str], hyp_tokens: List[str], n: int) -> Tuple[float, float, float]:
    """Computes precision, recall, and F1 for n-grams."""
    if not ref_tokens or not hyp_tokens:
        return 0.0, 0.0, 0.0
    ref_counts = compute_ngrams(ref_tokens, n)
    hyp_counts = compute_ngrams(hyp_tokens, n)

    total_ref = sum(ref_counts.values())
    total_hyp = sum(hyp_counts.values())
    if total_ref == 0 or total_hyp == 0:
        return 0.0, 0.0, 0.0

    overlap = sum(min(count, ref_counts.get(ng, 0)) for ng, count in hyp_counts.items())
    prec = overlap / total_hyp
    rec = overlap / total_ref
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def compute_lcs_length(tokens1: List[str], tokens2: List[str]) -> int:
    """Computes length of longest common subsequence (LCS)."""
    m, n = len(tokens1), len(tokens2)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]

def compute_rouge_l(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[float, float, float]:
    """Computes ROUGE-L precision, recall, and F1."""
    if not ref_tokens or not hyp_tokens:
        return 0.0, 0.0, 0.0
    lcs_len = compute_lcs_length(ref_tokens, hyp_tokens)
    prec = lcs_len / len(hyp_tokens) if hyp_tokens else 0.0
    rec = lcs_len / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def compute_bleu_1(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    """Computes sentence-level BLEU-1 with brevity penalty."""
    if not hyp_tokens:
        return 0.0
    prec, _, _ = compute_ngram_overlap(ref_tokens, hyp_tokens, n=1)
    if prec == 0.0:
        return 0.0
    c = len(hyp_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0
    return bp * prec

def compute_reply_lexical_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, Any]:
    """
    Computes lexical overlap metrics comparing agent replies against gold brand replies:
    - Mean ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1
    - Mean BLEU-1
    - Character and token length statistics
    """
    assert len(references) == len(hypotheses), f"Length mismatch: {len(references)} vs {len(hypotheses)}"
    if not references:
        return {}

    r1_f1s, r2_f1s, rl_f1s, bleu1s = [], [], [], []
    hyp_char_lens, hyp_token_lens = [], []
    ref_char_lens, ref_token_lens = [], []

    for ref, hyp in zip(references, hypotheses):
        ref_toks = tokenize_text(ref)
        hyp_toks = tokenize_text(hyp)

        _, _, r1 = compute_ngram_overlap(ref_toks, hyp_toks, n=1)
        _, _, r2 = compute_ngram_overlap(ref_toks, hyp_toks, n=2)
        _, _, rl = compute_rouge_l(ref_toks, hyp_toks)
        b1 = compute_bleu_1(ref_toks, hyp_toks)

        r1_f1s.append(r1)
        r2_f1s.append(r2)
        rl_f1s.append(rl)
        bleu1s.append(b1)

        hyp_char_lens.append(len(hyp))
        hyp_token_lens.append(len(hyp_toks))
        ref_char_lens.append(len(ref))
        ref_token_lens.append(len(ref_toks))

    n = len(references)
    return {
        "rouge_1_f1": round(sum(r1_f1s) / n, 4),
        "rouge_2_f1": round(sum(r2_f1s) / n, 4),
        "rouge_l_f1": round(sum(rl_f1s) / n, 4),
        "bleu_1": round(sum(bleu1s) / n, 4),
        "mean_hyp_char_length": round(sum(hyp_char_lens) / n, 1),
        "mean_hyp_token_length": round(sum(hyp_token_lens) / n, 1),
        "mean_ref_char_length": round(sum(ref_char_lens) / n, 1),
        "mean_ref_token_length": round(sum(ref_token_lens) / n, 1),
        "num_evaluated_replies": n
    }

def format_reply_metrics(metrics: Dict[str, Any]) -> str:
    """Formats reply generation lexical evaluation metrics into clean markdown."""
    lines = []
    lines.append(f"- **ROUGE-1 F1**: `{metrics['rouge_1_f1'] * 100:.2f}%`")
    lines.append(f"- **ROUGE-2 F1**: `{metrics['rouge_2_f1'] * 100:.2f}%`")
    lines.append(f"- **ROUGE-L F1**: `{metrics['rouge_l_f1'] * 100:.2f}%`")
    lines.append(f"- **BLEU-1**: `{metrics['bleu_1'] * 100:.2f}%`")
    lines.append(f"- **Mean Reply Length**: `{metrics['mean_hyp_token_length']} words` ({metrics['mean_hyp_char_length']} chars) vs Gold: `{metrics['mean_ref_token_length']} words` ({metrics['mean_ref_char_length']} chars)")
    return "\n".join(lines)
