"""
Offline Golden Evaluation Harness & Sanity-Check Baselines (Task T5).

Evaluates data/gold/gold_eval_200.jsonl against canonical standards:
- Validates data integrity, completeness, and holdout isolation.
- Computes Intent Accuracy, Macro-F1, and Per-Class Precision/Recall/F1.
- Computes Escalation Accuracy, Precision, Recall, and F1.
- Evaluates Majority-Intent and Always-Escalate dummy baselines.
- Generates structured evaluation reports in reports/ without altering gold data.

Execution:
    python -m scripts.eval_gold
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Ensure workspace root is on python sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.evaluation.validator import load_and_validate_gold, ALLOWED_INTENTS
from src.evaluation.metrics import (
    compute_multiclass_metrics,
    compute_binary_escalation_metrics,
    format_multiclass_report,
    format_confusion_matrix_ascii,
    format_binary_report,
)
from src.evaluation.baselines import MajorityIntentBaseline, AlwaysEscalateBaseline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Evaluation Harness for AppleSupport Golden Set (T5)"
    )
    parser.add_argument(
        "--gold-path",
        type=str,
        default="data/gold/gold_eval_200.jsonl",
        help="Path to labeled golden evaluation JSONL (default: data/gold/gold_eval_200.jsonl)"
    )
    parser.add_argument(
        "--holdout-path",
        type=str,
        default="data/gold/index_holdout_ids.txt",
        help="Path to holdout IDs file (default: data/gold/index_holdout_ids.txt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save evaluation report (default: reports/)"
    )
    return parser.parse_args()


def run_evaluation(gold_path: Path, holdout_path: Path, output_dir: Path) -> Dict[str, Any]:
    print("=" * 80)
    print(" TASK T5: GOLDEN EVALUATION HARNESS & SANITY-CHECK BASELINES")
    print("=" * 80)
    print(f"[*] Target Gold Dataset : {gold_path.resolve()}")
    print(f"[*] Holdout IDs File    : {holdout_path.resolve()}")
    print(f"[*] Output Directory    : {output_dir.resolve()}\n")

    # 1. Validation Step
    print("[1/4] Validating Golden Evaluation Dataset Integrity...")
    gold_records = load_and_validate_gold(
        gold_path=gold_path,
        holdout_ids_path=holdout_path,
        expected_count=200
    )
    print(f"  [+] PASSED: Verified exactly {len(gold_records)} rows.")
    print("  [+] PASSED: Zero duplicate thread IDs.")
    print("  [+] PASSED: All intents conform to canonical 10-class taxonomy.")
    print("  [+] PASSED: Escalation fields strictly typed (bool) with valid reason codes.")
    print("  [+] PASSED: Holdout IDs match line-for-line with index_holdout_ids.txt.\n")

    # Extract ground truth targets
    y_true_intent = [r["intent"] for r in gold_records]
    y_true_escalate = [r["escalate"] for r in gold_records]

    # 2. Baseline 1: Majority-Intent Baseline
    print("[2/4] Evaluating Baseline 1: Majority-Intent Classifier...")
    print("  (Sanity-check baseline: always predicts the most frequent intent in gold)")
    majority_model = MajorityIntentBaseline()
    majority_model.fit(gold_records)
    maj_intent = majority_model.majority_intent
    print(f"  [+] Identified Majority Intent: '{maj_intent}'")

    maj_preds = majority_model.predict(gold_records)
    y_pred_maj_intent = [p["predicted_intent"] for p in maj_preds]

    intent_metrics = compute_multiclass_metrics(
        y_true=y_true_intent,
        y_pred=y_pred_maj_intent,
        labels=ALLOWED_INTENTS
    )

    print("\n--- Majority-Intent Performance Report ---")
    print(format_multiclass_report(intent_metrics))
    print("\n--- Majority-Intent Confusion Matrix ---")
    print(format_confusion_matrix_ascii(intent_metrics["confusion_matrix"], ALLOWED_INTENTS))
    print()

    # 3. Baseline 2: Always-Escalate Baseline
    print("[3/4] Evaluating Baseline 2: Always-Escalate Triage Policy...")
    print("  (Sanity-check baseline: always routes every inquiry to human escalation)")
    escalate_model = AlwaysEscalateBaseline(default_intent=maj_intent)
    esc_preds = escalate_model.predict(gold_records)
    y_pred_always_esc = [p["predicted_escalate"] for p in esc_preds]

    escalate_metrics = compute_binary_escalation_metrics(
        y_true=y_true_escalate,
        y_pred=y_pred_always_esc
    )

    print("\n--- Always-Escalate Performance Report ---")
    print(format_binary_report(escalate_metrics))
    print()

    # 4. Summary & Report Serialization
    print("[4/4] Writing Standalone Evaluation Report (Zero Mutation to Gold JSONL)...")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = output_dir / "baseline_eval_results.json"
    report_md_path = output_dir / "baseline_eval_summary.md"

    results_payload = {
        "metadata": {
            "task": "T5_evaluation_harness_and_dummy_baselines",
            "gold_file": str(gold_path),
            "num_samples": len(gold_records),
            "holdout_isolated": True
        },
        "baseline_1_majority_intent": {
            "model_type": "MajorityIntentBaseline (Sanity Check)",
            "predicted_class": maj_intent,
            "overall_accuracy": intent_metrics["accuracy"],
            "macro_f1": intent_metrics["macro_f1"],
            "weighted_f1": intent_metrics["weighted_f1"],
            "per_class": intent_metrics["per_class"],
            "confusion_matrix": intent_metrics["confusion_matrix"]
        },
        "baseline_2_always_escalate": {
            "model_type": "AlwaysEscalateBaseline (Sanity Check)",
            "predicted_decision": True,
            "escalation_accuracy": escalate_metrics["accuracy"],
            "precision": escalate_metrics["precision"],
            "recall": escalate_metrics["recall"],
            "f1": escalate_metrics["f1"],
            "tp": escalate_metrics["tp"],
            "fp": escalate_metrics["fp"],
            "tn": escalate_metrics["tn"],
            "fn": escalate_metrics["fn"]
        }
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Task T5 Baseline Evaluation Summary\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write("This report documents the baseline performance established by non-learning dummy baselines on `data/gold/gold_eval_200.jsonl`.\n\n")
        f.write("| Baseline | Target Task | Accuracy | Macro-F1 | Precision | Recall | F1-Score |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        f.write(f"| **Majority Intent (`{maj_intent}`)** | Intent (10 classes) | **{intent_metrics['accuracy']*100:.2f}%** | **{intent_metrics['macro_f1']*100:.2f}%** | N/A | N/A | N/A |\n")
        f.write(f"| **Always Escalate (`True`)** | Escalation (Binary) | **{escalate_metrics['accuracy']*100:.2f}%** | N/A | **{escalate_metrics['precision']*100:.2f}%** | **{escalate_metrics['recall']*100:.2f}%** | **{escalate_metrics['f1']*100:.2f}%** |\n\n")
        f.write("## 2. Intent Classification Breakdown (Majority Intent)\n\n")
        f.write(format_multiclass_report(intent_metrics))
        f.write("\n\n```text\n")
        f.write(format_confusion_matrix_ascii(intent_metrics["confusion_matrix"], ALLOWED_INTENTS))
        f.write("\n```\n\n")
        f.write("## 3. Escalation Triage Breakdown (Always Escalate)\n\n")
        f.write(format_binary_report(escalate_metrics))
        f.write("\n")

    print(f"  [+] JSON Report saved to: {report_json_path.resolve()}")
    print(f"  [+] Markdown Summary saved to: {report_md_path.resolve()}")
    print("\n" + "=" * 80)
    print(" EVALUATION HARNESS EXECUTION COMPLETE")
    print("=" * 80 + "\n")

    return results_payload


def main():
    args = parse_args()
    gold_path = Path(args.gold_path)
    holdout_path = Path(args.holdout_path)
    output_dir = Path(args.output_dir)
    run_evaluation(gold_path, holdout_path, output_dir)


if __name__ == "__main__":
    main()
