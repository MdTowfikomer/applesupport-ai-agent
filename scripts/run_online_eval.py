"""
Online Golden Evaluation Runner (200 items with Gemini Flash LLM).

Evaluates the 200 holdout gold items using AppleSupportAgent(use_llm=True).
Caches predictions to:
- reports/online_eval_results_200.json
- online_eval_results_200.json (repo root for quick replay)
- reports/eval_results_agent_online.jsonl

Supports resumption, rate limit pacing, and detailed progress output.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.evaluation.validator import load_and_validate_gold, ALLOWED_INTENTS
from src.pipeline.agent import AppleSupportAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Run 200-item Online Evaluation with Gemini LLM")
    parser.add_argument(
        "--gold-path",
        type=str,
        default="data/gold/gold_eval_200.jsonl",
        help="Path to labeled golden evaluation JSONL"
    )
    parser.add_argument(
        "--holdout-path",
        type=str,
        default="data/gold/index_holdout_ids.txt",
        help="Path to holdout IDs file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="reports/online_eval_results_200.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Sleep delay in seconds between items to pace API calls (default: 0.35s)"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional max items to run (default: None, runs all 200)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resumption from existing cached predictions"
    )
    return parser.parse_args()


def run_online_eval():
    args = parse_args()
    gold_path = Path(args.gold_path)
    holdout_path = Path(args.holdout_path)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    root_output_file = WORKSPACE_ROOT / "online_eval_results_200.json"
    jsonl_output_file = output_file.parent / "eval_results_agent_online.jsonl"

    print("=" * 80)
    print(" ONLINE LLM EVALUATION RUNNER (200 GOLD HOLDOUT SAMPLES)")
    print("=" * 80)
    print(f"[*] Gold Dataset   : {gold_path}")
    print(f"[*] Output Target  : {output_file}")
    print(f"[*] Pacing Delay   : {args.delay}s per query")
    print(f"[*] Resumption     : {'Disabled' if args.no_resume else 'Enabled'}\n")

    # Load and validate gold set
    gold_records = load_and_validate_gold(
        gold_path=gold_path,
        holdout_ids_path=holdout_path,
        expected_count=200
    )
    if args.max_items:
        gold_records = gold_records[:args.max_items]

    # Check for existing cached predictions to resume
    cached_predictions: Dict[str, Dict[str, Any]] = {}
    if not args.no_resume and output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                items = data if isinstance(data, list) else data.get("predictions", [])
                for item in items:
                    if "thread_id" in item:
                        cached_predictions[item["thread_id"]] = item
            print(f"[*] Found {len(cached_predictions)} cached predictions in {output_file}. Resuming remaining...")
        except Exception as e:
            print(f"[!] Warning: Could not read existing cache ({e}). Starting fresh.")

    # Initialize LLM Agent
    print("[*] Initializing AppleSupportAgent with use_llm=True (Gemini Flash)...")
    agent = AppleSupportAgent(
        subsample_path="data/subsample/applesupport_threads_5k.jsonl",
        holdout_ids_path=str(holdout_path),
        use_llm=True
    )
    print(f"  [+] Pipeline model: {agent.model_name}")
    print(f"  [+] Retrieval corpus: {len(agent.retriever.corpus)} records (holdouts isolated)\n")

    results: List[Dict[str, Any]] = []
    start_total_time = time.time()
    correct_intents = 0
    correct_escalations = 0

    for idx, gold in enumerate(gold_records):
        tid = gold["thread_id"]
        
        # If already cached and valid, reuse
        if tid in cached_predictions and cached_predictions[tid].get("predicted_intent") and cached_predictions[tid].get("predicted_reply"):
            pred = cached_predictions[tid]
            results.append(pred)
            is_match = (pred["predicted_intent"] == gold["intent"])
            esc_match = (pred["predicted_escalate"] == gold["escalate"])
            if is_match:
                correct_intents += 1
            if esc_match:
                correct_escalations += 1
            print(f"[{idx+1:03d}/{len(gold_records)}] [CACHED] tid={tid} | Gold: {gold['intent']:<26} -> Pred: {pred['predicted_intent']:<26} ({'MATCH' if is_match else 'DIFF'})")
            continue

        # Execute agent prediction
        t0 = time.time()
        agent_out = agent.predict_record(gold)
        dt = time.time() - t0

        pred_record = {
            "thread_id": tid,
            "customer_text": gold["customer_text"],
            "gold_intent": gold["intent"],
            "predicted_intent": agent_out["predicted_intent"],
            "gold_escalate": gold["escalate"],
            "predicted_escalate": agent_out["predicted_escalate"],
            "gold_escalate_reason": gold.get("escalate_reason"),
            "predicted_escalate_reason": agent_out.get("predicted_escalate_reason"),
            "gold_reply": gold["brand_text"],
            "predicted_reply": agent_out["predicted_reply"],
            "retrieved_thread_id": agent_out.get("retrieved_thread_id"),
            "retrieval_score": agent_out.get("retrieval_score"),
            "pipeline_model": agent_out.get("pipeline_model"),
        }
        results.append(pred_record)
        cached_predictions[tid] = pred_record

        is_match = (agent_out["predicted_intent"] == gold["intent"])
        esc_match = (agent_out["predicted_escalate"] == gold["escalate"])
        if is_match:
            correct_intents += 1
        if esc_match:
            correct_escalations += 1

        curr_acc = correct_intents / len(results)
        print(
            f"[{idx+1:03d}/{len(gold_records)}] ({dt:.2f}s) tid={tid} | "
            f"Gold: {gold['intent']:<26} -> Pred: {agent_out['predicted_intent']:<26} "
            f"({'MATCH' if is_match else 'DIFF'}) | RunAcc: {curr_acc*100:.1f}%"
        )

        # Checkpoint every 10 items or on completion
        if (idx + 1) % 10 == 0 or (idx + 1) == len(gold_records):
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        if args.delay > 0 and (idx + 1) < len(gold_records):
            time.sleep(args.delay)

    total_time = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f" ONLINE EVALUATION FINISHED in {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f" Final Raw Intent Accuracy : {correct_intents}/{len(results)} ({correct_intents/len(results)*100:.2f}%)")
    print(f" Final Raw Escalation Match: {correct_escalations}/{len(results)} ({correct_escalations/len(results)*100:.2f}%)")
    print("=" * 80)

    # Write final outputs
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved full results to: {output_file.resolve()}")

    with open(root_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved root copy to    : {root_output_file.resolve()}")

    with open(jsonl_output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[+] Saved jsonl lines to  : {jsonl_output_file.resolve()}")


if __name__ == "__main__":
    run_online_eval()
