import json

with open("data/gold/gold_eval_200.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

with open("data/gold/labeled_summary.txt", "w", encoding="utf-8") as out:
    for i, r in enumerate(records):
        it = r.get("intent")
        esc = r.get("escalate")
        rea = r.get("escalate_reason")
        status = "LABELED" if it is not None else "EMPTY"
        out.write(f"Row {i+1:03d} [{r['thread_id']}] [{status}] intent={it}, esc={esc}, reason={rea}\n")
        out.write(f"  Cust:  {r['customer_text'].replace(chr(10), ' ')}\n")
        out.write(f"  Brand: {r['brand_text'].replace(chr(10), ' ')}\n")
        out.write("-" * 80 + "\n")

print("Dumped labeled_summary.txt successfully.")
