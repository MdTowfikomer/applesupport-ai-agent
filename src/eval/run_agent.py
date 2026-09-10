"""
Interactive CLI Runner for AppleSupport AI Agent (Task T8).
Allows processing:
1. Single query:
   python -m src.eval.run_agent --text "My phone battery drains in 2 hours on iOS 11.1"
2. Evaluating sample gold records:
   python -m src.eval.run_agent --samples 5
3. Running offline test mode:
   python -m src.eval.run_agent --offline --text "Screen is cracked"
"""

import sys
import json
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.pipeline.agent import AppleSupportAgent


def main():
    parser = argparse.ArgumentParser(description="AppleSupport AI Agent Runner (Task T8)")
    parser.add_argument("--text", type=str, default="", help="Single customer tweet text to process")
    parser.add_argument("--samples", type=int, default=0, help="Number of gold samples to run (0 for none)")
    parser.add_argument("--offline", action="store_true", help="Run in deterministic offline mode without calling Gemini API")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidate resolutions to retrieve (default: 3)")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model name")
    args = parser.parse_args()

    agent = AppleSupportAgent(
        subsample_path="data/subsample/applesupport_threads_5k.jsonl",
        holdout_ids_path="data/gold/index_holdout_ids.txt",
        use_llm=not args.offline,
        top_k=args.top_k,
        model_name=args.model,
    )

    if args.text:
        print("\n" + "=" * 70)
        print(" APPLESUPPORT AI AGENT EXECUTION")
        print("=" * 70)
        print(f"Customer Tweet : \"{args.text}\"")
        res = agent.process(args.text)
        print(f"Intent         : {res['predicted_intent']}")
        print(f"Escalate       : {res['predicted_escalate']}")
        print(f"Reason Code    : {res['predicted_escalate_reason']}")
        print(f"Drafted Reply  : \"{res['predicted_reply']}\"")
        print(f"\nTop Retrieved Historical Resolution (Thread: {res['retrieved_thread_id']}, Score: {res['retrieval_score']}):")
        if res["retrieved_candidates"]:
            cand = res["retrieved_candidates"][0]
            print(f"  Customer: \"{cand['customer_text']}\"")
            print(f"  Brand   : \"{cand['brand_text']}\"")
        print("=" * 70 + "\n")

    elif args.samples > 0:
        gold_path = Path("data/gold/gold_eval_200.jsonl")
        with open(gold_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f][:args.samples]

        print(f"\nRunning {len(records)} sample records from {gold_path}...\n")
        for i, rec in enumerate(records, 1):
            res = agent.process(rec["customer_text"], thread_id=rec["thread_id"])
            print(f"[{i}/{len(records)}] Thread: {rec['thread_id']}")
            print(f"  Query        : {rec['customer_text'][:70]}...")
            print(f"  Gold Intent  : {rec['intent']:<30} | Pred Intent : {res['predicted_intent']}")
            print(f"  Gold Escalate: {str(rec['escalate']):<30} | Pred Escalate: {str(res['predicted_escalate'])} ({res['predicted_escalate_reason']})")
            print(f"  Drafted Reply: \"{res['predicted_reply'][:80]}...\"")
            print("-" * 70)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
