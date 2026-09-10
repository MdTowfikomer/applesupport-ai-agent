"""
CLI runner for baseline agents (Task T6).
Allows invoking:
    python -m src.eval.run_baseline --type trivial
"""

import sys
import argparse
from pathlib import Path

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.eval_gold import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations on AppleSupport gold set")
    parser.add_argument(
        "--type",
        type=str,
        default="simple",
        choices=["simple", "trivial", "majority_intent", "always_escalate", "all"],
        help="Baseline type to run (default: simple)"
    )
    parser.add_argument(
        "--gold-path",
        type=str,
        default="data/gold/gold_eval_200.jsonl",
        help="Path to gold evaluation set"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output reports directory"
    )
    args = parser.parse_args()

    run_evaluation(
        gold_path=Path(args.gold_path),
        holdout_path=Path("data/gold/index_holdout_ids.txt"),
        output_dir=Path(args.output_dir),
        baseline=args.type
    )


if __name__ == "__main__":
    main()
