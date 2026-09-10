"""
Offline Golden Evaluation Harness & Baseline Benchmarks (Tasks T5 & T6).

Evaluates data/gold/gold_eval_200.jsonl against canonical standards:
- Validates data integrity, completeness, and holdout isolation.
- Computes Intent Accuracy, Macro-F1, and Per-Class Precision/Recall/F1.
- Computes Escalation Accuracy, Precision, Recall, and F1.
- Computes Response Lexical Metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU-1, Length).
- Evaluates the End-to-End Trivial Baseline Agent and component baselines.
- Generates structured evaluation reports in reports/ without altering gold data.

Execution:
    python -m scripts.eval_gold
    python -m scripts.eval_gold --baseline trivial
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure workspace root is on python sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.evaluation.validator import load_and_validate_gold, ALLOWED_INTENTS
from src.evaluation.metrics import (
    compute_multiclass_metrics,
    compute_binary_escalation_metrics,
    compute_reply_lexical_metrics,
    format_multiclass_report,
    format_confusion_matrix_ascii,
    format_binary_report,
    format_reply_metrics,
)
from src.evaluation.baselines import (
    MajorityIntentBaseline,
    AlwaysEscalateBaseline,
    TrivialBaselineAgent,
)
from src.baselines.simple import SimpleBaselineAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Evaluation Harness for AppleSupport Golden Set (Tasks T5, T6, T7)"
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
    parser.add_argument(
        "--baseline",
        type=str,
        default="all",
        choices=["all", "simple", "trivial", "majority_intent", "always_escalate"],
        help="Which baseline to evaluate (default: all)"
    )
    return parser.parse_args()


def run_evaluation(
    gold_path: Path,
    holdout_path: Path,
    output_dir: Path,
    baseline: str = "all"
) -> Dict[str, Any]:
    print("=" * 80)
    print(" TASK T5 / T6: GOLDEN EVALUATION HARNESS & BASELINE BENCHMARKS")
    print("=" * 80)
    print(f"[*] Target Gold Dataset : {gold_path.resolve()}")
    print(f"[*] Holdout IDs File    : {holdout_path.resolve()}")
    print(f"[*] Output Directory    : {output_dir.resolve()}")
    print(f"[*] Evaluation Target   : baseline='{baseline}'\n")

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
    gold_replies = [r["brand_text"] for r in gold_records]

    results_payload: Dict[str, Any] = {
        "metadata": {
            "task": "T5_T6_evaluation_harness_and_baselines",
            "gold_file": str(gold_path),
            "num_samples": len(gold_records),
            "holdout_isolated": True,
            "baseline_evaluated": baseline
        }
    }

    # 2. Trivial Baseline Agent (T6)
    if baseline in ["all", "trivial"]:
        print("[2/4] Evaluating Trivial Baseline Agent (Task T6)...")
        print("  (End-to-End Baseline: Majority Intent + Majority Escalation + Canned Reply)")
        trivial_agent = TrivialBaselineAgent()
        trivial_agent.fit(gold_records)
        
        trivial_preds = trivial_agent.predict(gold_records)
        y_pred_triv_intent = [p["predicted_intent"] for p in trivial_preds]
        y_pred_triv_escalate = [p["predicted_escalate"] for p in trivial_preds]
        hyp_replies = [p["predicted_reply"] for p in trivial_preds]

        # Compute intent metrics
        triv_intent_metrics = compute_multiclass_metrics(
            y_true=y_true_intent,
            y_pred=y_pred_triv_intent,
            labels=ALLOWED_INTENTS
        )

        # Compute escalation metrics
        triv_escalate_metrics = compute_binary_escalation_metrics(
            y_true=y_true_escalate,
            y_pred=y_pred_triv_escalate
        )

        # Compute reply lexical metrics
        triv_reply_metrics = compute_reply_lexical_metrics(
            references=gold_replies,
            hypotheses=hyp_replies
        )

        print("\n--- [T6] Trivial Baseline: Intent Classification ---")
        print(format_multiclass_report(triv_intent_metrics))
        print("\n--- [T6] Trivial Baseline: Intent Confusion Matrix ---")
        print(format_confusion_matrix_ascii(triv_intent_metrics["confusion_matrix"], ALLOWED_INTENTS))
        print("\n--- [T6] Trivial Baseline: Escalation Triage ---")
        print(format_binary_report(triv_escalate_metrics))
        print("\n--- [T6] Trivial Baseline: Reply Generation Lexical Overlap ---")
        print(format_reply_metrics(triv_reply_metrics))
        print()

        # Save predictions JSONL
        output_dir.mkdir(parents=True, exist_ok=True)
        triv_jsonl_path = output_dir / "eval_results_trivial.jsonl"
        with open(triv_jsonl_path, "w", encoding="utf-8") as f:
            for g, p in zip(gold_records, trivial_preds):
                item = {
                    "thread_id": g["thread_id"],
                    "customer_text": g["customer_text"],
                    "gold_intent": g["intent"],
                    "predicted_intent": p["predicted_intent"],
                    "gold_escalate": g["escalate"],
                    "predicted_escalate": p["predicted_escalate"],
                    "gold_escalate_reason": g.get("escalate_reason"),
                    "predicted_escalate_reason": p.get("predicted_escalate_reason"),
                    "gold_reply": g["brand_text"],
                    "predicted_reply": p["predicted_reply"]
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  [+] Saved per-sample predictions to: {triv_jsonl_path.resolve()}")

        results_payload["trivial_baseline_agent"] = {
            "model_type": "TrivialBaselineAgent",
            "majority_intent": trivial_agent.majority_intent,
            "escalate_decision": trivial_agent.escalate,
            "escalate_reason": trivial_agent.escalate_reason,
            "canned_reply": trivial_agent.canned_reply,
            "intent_metrics": {
                "accuracy": triv_intent_metrics["accuracy"],
                "macro_f1": triv_intent_metrics["macro_f1"],
                "weighted_f1": triv_intent_metrics["weighted_f1"],
                "per_class": triv_intent_metrics["per_class"]
            },
            "escalation_metrics": triv_escalate_metrics,
            "reply_metrics": triv_reply_metrics
        }

    # 3. Simple Baseline Agent (T7)
    if baseline in ["all", "simple"]:
        print("[2b/4] Evaluating Simple Baseline Agent (Task T7)...")
        print("  (TF-IDF Retrieval + Keyword Intent + Rule-Based Escalation)")
        simple_agent = SimpleBaselineAgent(
            subsample_path="data/subsample/applesupport_threads_5k.jsonl",
            holdout_ids_path=str(holdout_path)
        )
        # Verify strict holdout isolation
        assert len(simple_agent.holdout_ids) == 200, f"Expected 200 holdouts, got {len(simple_agent.holdout_ids)}"
        assert len(simple_agent.corpus) == 4800, f"Expected 4800 non-holdouts, got {len(simple_agent.corpus)}"
        assert not any(r["thread_id"] in simple_agent.holdout_ids for r in simple_agent.corpus), "Data leakage detected!"

        simple_preds = simple_agent.predict(gold_records)
        y_pred_simple_intent = [p["predicted_intent"] for p in simple_preds]
        y_pred_simple_escalate = [p["predicted_escalate"] for p in simple_preds]
        hyp_simple_replies = [p["predicted_reply"] for p in simple_preds]

        # Compute intent metrics
        simple_intent_metrics = compute_multiclass_metrics(
            y_true=y_true_intent,
            y_pred=y_pred_simple_intent,
            labels=ALLOWED_INTENTS
        )

        # Compute escalation metrics
        simple_escalate_metrics = compute_binary_escalation_metrics(
            y_true=y_true_escalate,
            y_pred=y_pred_simple_escalate
        )

        # Compute reply lexical metrics
        simple_reply_metrics = compute_reply_lexical_metrics(
            references=gold_replies,
            hypotheses=hyp_simple_replies
        )

        print("\n--- [T7] Simple Baseline: Intent Classification ---")
        print(format_multiclass_report(simple_intent_metrics))
        print("\n--- [T7] Simple Baseline: Intent Confusion Matrix ---")
        print(format_confusion_matrix_ascii(simple_intent_metrics["confusion_matrix"], ALLOWED_INTENTS))
        print("\n--- [T7] Simple Baseline: Escalation Triage ---")
        print(format_binary_report(simple_escalate_metrics))
        print("\n--- [T7] Simple Baseline: Reply Generation Lexical Overlap ---")
        print(format_reply_metrics(simple_reply_metrics))
        print()

        # Save predictions JSONL
        output_dir.mkdir(parents=True, exist_ok=True)
        simple_jsonl_path = output_dir / "eval_results_simple.jsonl"
        with open(simple_jsonl_path, "w", encoding="utf-8") as f:
            for g, p in zip(gold_records, simple_preds):
                item = {
                    "thread_id": g["thread_id"],
                    "customer_text": g["customer_text"],
                    "gold_intent": g["intent"],
                    "predicted_intent": p["predicted_intent"],
                    "gold_escalate": g["escalate"],
                    "predicted_escalate": p["predicted_escalate"],
                    "gold_escalate_reason": g.get("escalate_reason"),
                    "predicted_escalate_reason": p.get("predicted_escalate_reason"),
                    "gold_reply": g["brand_text"],
                    "predicted_reply": p["predicted_reply"],
                    "retrieved_thread_id": p.get("retrieved_thread_id"),
                    "retrieval_score": p.get("retrieval_score")
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  [+] Saved per-sample predictions to: {simple_jsonl_path.resolve()}")

        results_payload["simple_baseline_agent"] = {
            "model_type": "SimpleBaselineAgent",
            "retrieval_corpus_size": len(simple_agent.corpus),
            "holdout_ids_excluded": len(simple_agent.holdout_ids),
            "intent_metrics": {
                "accuracy": simple_intent_metrics["accuracy"],
                "macro_f1": simple_intent_metrics["macro_f1"],
                "weighted_f1": simple_intent_metrics["weighted_f1"],
                "per_class": simple_intent_metrics["per_class"]
            },
            "escalation_metrics": simple_escalate_metrics,
            "reply_metrics": simple_reply_metrics
        }

    # 4. Individual Dummy Baselines (T5)
    if baseline in ["all", "majority_intent"]:
        print("[3/4] Evaluating Component Baseline: Majority-Intent Classifier...")
        majority_model = MajorityIntentBaseline().fit(gold_records)
        maj_preds = majority_model.predict(gold_records)
        y_pred_maj_intent = [p["predicted_intent"] for p in maj_preds]
        maj_intent_metrics = compute_multiclass_metrics(
            y_true=y_true_intent,
            y_pred=y_pred_maj_intent,
            labels=ALLOWED_INTENTS
        )
        results_payload["majority_intent_baseline"] = {
            "predicted_class": majority_model.majority_intent,
            "accuracy": maj_intent_metrics["accuracy"],
            "macro_f1": maj_intent_metrics["macro_f1"],
            "weighted_f1": maj_intent_metrics["weighted_f1"]
        }

    if baseline in ["all", "always_escalate"]:
        print("[3/4] Evaluating Component Baseline: Always-Escalate Policy...")
        escalate_model = AlwaysEscalateBaseline()
        esc_preds = escalate_model.predict(gold_records)
        y_pred_always_esc = [p["predicted_escalate"] for p in esc_preds]
        always_esc_metrics = compute_binary_escalation_metrics(
            y_true=y_true_escalate,
            y_pred=y_pred_always_esc
        )
        results_payload["always_escalate_baseline"] = always_esc_metrics

    # 5. Summary & Report Serialization
    print("\n[4/4] Serializing Final Benchmark Reports (Zero Mutation to Gold JSONL)...")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = output_dir / "baseline_eval_results.json"
    report_md_path = output_dir / "baseline_eval_summary.md"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# AppleSupport Baseline Evaluation Summary (Tasks T5, T6 & T7)\n\n")
        f.write("## 1. Executive Summary & Benchmark Floor\n\n")
        f.write("This report documents empirical baseline performance established on `data/gold/gold_eval_200.jsonl`.\n\n")
        
        f.write("| Baseline Pipeline | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Escalation Precision | Escalation Recall | Escalation F1 | ROUGE-1 F1 | ROUGE-L F1 | BLEU-1 |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        if "simple_baseline_agent" in results_payload:
            sb = results_payload["simple_baseline_agent"]
            im = sb["intent_metrics"]
            em = sb["escalation_metrics"]
            rm = sb["reply_metrics"]
            f.write(f"| **Simple Baseline Agent (T7)** | **{im['accuracy']*100:.2f}%** | **{im['macro_f1']*100:.2f}%** | **{em['accuracy']*100:.2f}%** | **{em['precision']*100:.2f}%** | **{em['recall']*100:.2f}%** | **{em['f1']*100:.2f}%** | **{rm['rouge_1_f1']*100:.2f}%** | **{rm['rouge_l_f1']*100:.2f}%** | **{rm['bleu_1']*100:.2f}%** |\n")
        if "trivial_baseline_agent" in results_payload:
            tb = results_payload["trivial_baseline_agent"]
            im = tb["intent_metrics"]
            em = tb["escalation_metrics"]
            rm = tb["reply_metrics"]
            f.write(f"| **Trivial Baseline Agent (T6)** | {im['accuracy']*100:.2f}% | {im['macro_f1']*100:.2f}% | {em['accuracy']*100:.2f}% | {em['precision']*100:.2f}% | {em['recall']*100:.2f}% | {em['f1']*100:.2f}% | {rm['rouge_1_f1']*100:.2f}% | {rm['rouge_l_f1']*100:.2f}% | {rm['bleu_1']*100:.2f}% |\n")
        if "majority_intent_baseline" in results_payload:
            mi = results_payload["majority_intent_baseline"]
            f.write(f"| *Majority-Intent Only (T5)* | {mi['accuracy']*100:.2f}% | {mi['macro_f1']*100:.2f}% | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n")
        if "always_escalate_baseline" in results_payload:
            ae = results_payload["always_escalate_baseline"]
            f.write(f"| *Always-Escalate Only (T5)* | N/A | N/A | {ae['accuracy']*100:.2f}% | {ae['precision']*100:.2f}% | {ae['recall']*100:.2f}% | {ae['f1']*100:.2f}% | N/A | N/A | N/A |\n")
        
        f.write("\n## 2. Simple Baseline Agent (Task T7)\n\n")
        if "simple_baseline_agent" in results_payload:
            sb = results_payload["simple_baseline_agent"]
            f.write(f"- **Intent Classifier**: Keyword & regex pattern matcher adhering to codebook hierarchy (Rule 3: account > billing > hardware > symptoms > vague).\n")
            f.write(f"- **Escalation Triage**: Deterministic safety, legal, credential, billing, and screenshot-only trigger rules.\n")
            f.write(f"- **Resolution Retrieval**: TF-IDF cosine-similarity retriever indexed over **{sb['retrieval_corpus_size']}** historical non-holdout threads (strictly excluding all {sb['holdout_ids_excluded']} holdouts).\n\n")
            f.write("### Intent Classification Breakdown\n\n")
            f.write(format_multiclass_report(simple_intent_metrics))
            f.write("\n\n```text\n")
            f.write(format_confusion_matrix_ascii(simple_intent_metrics["confusion_matrix"], ALLOWED_INTENTS))
            f.write("\n```\n\n")
            f.write("### Escalation Triage Breakdown\n\n")
            f.write(format_binary_report(simple_escalate_metrics))
            f.write("\n\n### Reply Generation Lexical Overlap\n\n")
            f.write(format_reply_metrics(simple_reply_metrics))
            f.write("\n\n")

        f.write("## 3. Trivial Baseline Agent Configuration (Task T6)\n\n")
        if "trivial_baseline_agent" in results_payload:
            tb = results_payload["trivial_baseline_agent"]
            f.write(f"- **Intent Decision**: Fixed majority class (`{tb['majority_intent']}`)\n")
            f.write(f"- **Escalation Routing**: Fixed majority decision (`escalate = {tb['escalate_decision']}`, reason = `{tb['escalate_reason']}`)\n")
            f.write(f"- **Canned Reply Template**: `\"{tb['canned_reply']}\"`\n\n")
            f.write("### Intent Classification Breakdown\n\n")
            f.write(format_multiclass_report(triv_intent_metrics))
            f.write("\n\n```text\n")
            f.write(format_confusion_matrix_ascii(triv_intent_metrics["confusion_matrix"], ALLOWED_INTENTS))
            f.write("\n```\n\n")
            f.write("### Escalation Triage Breakdown\n\n")
            f.write(format_binary_report(triv_escalate_metrics))
            f.write("\n\n### Reply Generation Lexical Overlap\n\n")
            f.write(format_reply_metrics(triv_reply_metrics))
            f.write("\n")

    print(f"  [+] JSON Results saved to : {report_json_path.resolve()}")
    print(f"  [+] Markdown Summary saved: {report_md_path.resolve()}")
    print("\n" + "=" * 80)
    print(" EVALUATION HARNESS EXECUTION COMPLETE")
    print("=" * 80 + "\n")

    return results_payload


def main():
    args = parse_args()
    gold_path = Path(args.gold_path)
    holdout_path = Path(args.holdout_path)
    output_dir = Path(args.output_dir)
    run_evaluation(gold_path, holdout_path, output_dir, baseline=args.baseline)


if __name__ == "__main__":
    main()
