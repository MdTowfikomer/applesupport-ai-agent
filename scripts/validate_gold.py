import json

with open("data/gold/schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

valid_intents = set(schema["properties"]["intent"]["enum"])
valid_reasons = set(schema["properties"]["escalate_reason"]["enum"])

with open("data/gold/gold_eval_200.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

assert len(records) == 200, f"Expected 200 records, got {len(records)}"

errors = []
for idx, r in enumerate(records):
    # Check required fields
    for field in ["thread_id", "customer_tweet_id", "customer_text", "brand_text", "intent", "escalate"]:
        if field not in r:
            errors.append(f"Row {idx+1} missing required field {field}")
            
    # Check intent enum
    if r.get("intent") not in valid_intents or r.get("intent") is None:
        errors.append(f"Row {idx+1} has invalid intent: {r.get('intent')}")
        
    # Check escalate boolean
    if not isinstance(r.get("escalate"), bool):
        errors.append(f"Row {idx+1} has non-bool escalate: {r.get('escalate')}")
        
    # Check escalate reason consistency
    if r.get("escalate") is True:
        if r.get("escalate_reason") not in valid_reasons or not r.get("escalate_reason"):
            errors.append(f"Row {idx+1} escalate=True but reason is invalid or empty: {r.get('escalate_reason')}")
    else:
        if r.get("escalate_reason") is not None and r.get("escalate_reason") != "":
            errors.append(f"Row {idx+1} escalate=False but reason is not null/empty: {r.get('escalate_reason')}")

if errors:
    print(f"Validation failed with {len(errors)} errors:")
    for e in errors[:10]:
        print("  ", e)
    raise SystemExit(1)
else:
    print("SUCCESS: All 200 rows strictly pass schema validation!")
