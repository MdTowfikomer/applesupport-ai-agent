import json

with open("data/gold/gold_eval_200.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

total = len(records)
labeled = [i for i, r in enumerate(records) if r.get("intent") is not None]
unlabeled = [i for i, r in enumerate(records) if r.get("intent") is None]

print(f"Total rows: {total}")
print(f"Labeled rows: {len(labeled)}")
print(f"Unlabeled rows: {len(unlabeled)}")
print(f"Labeled indices range: {min(labeled) if labeled else 'None'} to {max(labeled) if labeled else 'None'}")

# Distribution of labels so far
intent_dist = {}
escalate_dist = {}
reason_dist = {}

for i in labeled:
    r = records[i]
    it = r.get("intent")
    esc = r.get("escalate")
    rea = r.get("escalate_reason")
    intent_dist[it] = intent_dist.get(it, 0) + 1
    escalate_dist[esc] = escalate_dist.get(esc, 0) + 1
    reason_dist[rea] = reason_dist.get(rea, 0) + 1

print("\n--- Current Intent Distribution ---")
for k, v in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v}")

print("\n--- Current Escalate Distribution ---")
for k, v in sorted(escalate_dist.items(), key=lambda x: str(x[0])):
    print(f"  {k}: {v}")

print("\n--- Current Reason Distribution ---")
for k, v in sorted(reason_dist.items(), key=lambda x: str(x[0])):
    print(f"  {k}: {v}")

print("\n--- Sample of labeled rows ---")
for i in labeled[:10]:
    r = records[i]
    print(f"Row {i+1} [{r['thread_id']}]: intent={r['intent']}, esc={r['escalate']}, reason={r['escalate_reason']}")
    print(f"  Cust: {r['customer_text'][:90]}")
    print(f"  Brand: {r['brand_text'][:90]}")
