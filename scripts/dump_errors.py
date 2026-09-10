"""
CLI Script to extract, dump, and categorize prediction errors for AppleSupport AI Agent (Task T10).

Usage:
    python -m scripts.dump_errors
    python -m scripts.dump_errors --predictions reports/eval_results_agent.jsonl --output-dir reports
"""

import sys
import argparse
from pathlib import Path

# Windows console encoding fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.evaluation.error_analyzer import dump_errors_and_summarize


def main():
    parser = argparse.ArgumentParser(description="Systematic Error Analysis & Failure Mode Dump (Task T10)")
    parser.add_argument(
        "--predictions",
        type=str,
        default="reports/eval_results_agent.jsonl",
        help="Path to evaluation predictions JSONL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save error dump and summary"
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print(" TASK T10: SYSTEMATIC ERROR ANALYSIS & FAILURE TAXONOMY DUMP")
    print("=" * 80)
    print(f"[*] Input Predictions  : {pred_path.resolve()}")
    print(f"[*] Output Directory   : {output_dir.resolve()}\n")

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found at: {pred_path}. Run 'python -m scripts.eval_gold' first.")

    res = dump_errors_and_summarize(
        predictions_path=pred_path,
        output_dir=output_dir
    )

    print("=" * 80)
    print(" ERROR ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"- Total Error Cases Dumped : {res['total_errors']}")
    print(f"- Intent Classification Errs: {res['intent_error_count']}")
    print(f"- Escalation Triage Errs   : {res['escalation_error_count']}")
    print(f"\n[+] Machine-Readable Error Dump : {res['dump_path'].resolve()}")
    print(f"[+] Human-Readable Markdown     : {res['summary_path'].resolve()}")
    print(f"[+] Summary Error Metrics       : {res['metrics_path'].resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
