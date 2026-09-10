"""
Inter-Annotator & Judge Agreement Metrics Engine (Task T9).
Calculates statistical agreement between Human Annotators and the LLM-as-a-Judge:
1. Pearson Correlation (r)
2. Spearman Rank Correlation (rho)
3. Cohen's Quadratic Weighted Kappa (QWK)
4. Exact Match Agreement (%)
5. Within-1 Point Tolerance Agreement (%)
6. Mean Absolute Error (MAE)
"""

import math
from typing import List, Dict, Any, Tuple


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Computes Pearson product-moment correlation coefficient."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0
    return max(-1.0, min(1.0, cov / denom))


def _rank(vals: List[float]) -> List[float]:
    """Computes fractional ranks for Spearman correlation with ties handling."""
    n = len(vals)
    indexed = sorted(enumerate(vals), key=lambda item: item[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = 1.0 + (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: List[float], y: List[float]) -> float:
    """Computes Spearman rank-order correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    return pearson_correlation(rx, ry)


def quadratic_weighted_kappa(
    y_true: List[int],
    y_pred: List[int],
    min_rating: int = 1,
    max_rating: int = 5,
) -> float:
    """
    Computes Cohen's Quadratic Weighted Kappa (QWK) for ordinal ratings.
    Penalizes disagreements proportionally to the squared distance between scores.
    """
    n = len(y_true)
    if n != len(y_pred) or n == 0:
        return 0.0

    num_classes = max_rating - min_rating + 1
    if num_classes <= 1:
        return 1.0

    # Build confusion matrix O (observed)
    observed = [[0.0] * num_classes for _ in range(num_classes)]
    hist_true = [0.0] * num_classes
    hist_pred = [0.0] * num_classes

    for yt, yp in zip(y_true, y_pred):
        # Clip to rating bounds
        idx_t = max(0, min(num_classes - 1, yt - min_rating))
        idx_p = max(0, min(num_classes - 1, yp - min_rating))
        observed[idx_t][idx_p] += 1.0
        hist_true[idx_t] += 1.0
        hist_pred[idx_p] += 1.0

    # Build expected matrix E
    expected = [[0.0] * num_classes for _ in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            expected[i][j] = (hist_true[i] * hist_pred[j]) / n

    # Weights matrix (quadratic)
    denominator = (num_classes - 1) ** 2
    sum_w_obs = 0.0
    sum_w_exp = 0.0

    for i in range(num_classes):
        for j in range(num_classes):
            w = ((i - j) ** 2) / denominator
            sum_w_obs += w * observed[i][j]
            sum_w_exp += w * expected[i][j]

    if sum_w_exp == 0.0:
        return 1.0 if sum_w_obs == 0.0 else 0.0

    return 1.0 - (sum_w_obs / sum_w_exp)


def exact_match_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Computes exact match agreement percentage."""
    if not y_true:
        return 0.0
    return sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)


def within_one_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Computes percentage of ratings agreeing within 1 point tolerance."""
    if not y_true:
        return 0.0
    return sum(1 for yt, yp in zip(y_true, y_pred) if abs(yt - yp) <= 1) / len(y_true)


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """Computes Mean Absolute Error between scores."""
    if not y_true:
        return 0.0
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)


def compute_agreement_report(
    human_scores: List[int],
    judge_scores: List[int],
    dimension_name: str = "overall_quality",
) -> Dict[str, Any]:
    """Computes complete agreement battery for an evaluation dimension."""
    pearson = pearson_correlation([float(s) for s in human_scores], [float(s) for s in judge_scores])
    spearman = spearman_correlation([float(s) for s in human_scores], [float(s) for s in judge_scores])
    qwk = quadratic_weighted_kappa(human_scores, judge_scores, min_rating=1, max_rating=5)
    exact = exact_match_accuracy(human_scores, judge_scores)
    within_1 = within_one_accuracy(human_scores, judge_scores)
    mae = mean_absolute_error([float(s) for s in human_scores], [float(s) for s in judge_scores])

    return {
        "dimension": dimension_name,
        "sample_size": len(human_scores),
        "pearson_r": round(pearson, 4),
        "spearman_rho": round(spearman, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "exact_match_pct": round(exact * 100, 2),
        "within_one_pct": round(within_1 * 100, 2),
        "mean_absolute_error": round(mae, 4),
    }
