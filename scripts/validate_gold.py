import json
from pathlib import Path

# 1. Load schema
with open("data/gold/schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

valid_intents = set(schema["properties"]["intent"]["enum"])
valid_reasons = set(schema["properties"]["escalate_reason"]["enum"])

# 2. Load eval records
with open("data/gold/gold_eval_200.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

assert len(records) == 200, f"Expected 200 records, got {len(records)}"

# 3. Load subsample for source fidelity check
with open("data/subsample/applesupport_threads_5k.jsonl", "r", encoding="utf-8") as f:
    subsample = {json.loads(line)["thread_id"]: json.loads(line) for line in f}

# 4. Load holdout IDs
with open("data/gold/index_holdout_ids.txt", "r", encoding="utf-8") as f:
    holdout_ids = [line.strip() for line in f if line.strip()]

assert len(holdout_ids) == 200, f"Expected 200 holdout IDs, got {len(holdout_ids)}"
assert [r["thread_id"] for r in records] == holdout_ids, "Holdout IDs do not match gold_eval_200.jsonl line-for-line!"

# Mandatory escalation intents per docs/intent_codebook.md §5.2
mandatory_escalation_intents = {
    "account_access_security",
    "billing_purchases_subscriptions",
    "hardware_physical_accessory"
}

errors = []
for idx, r in enumerate(records):
    tid = r.get("thread_id")
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

    # Check codebook §5.2 mandatory escalation rule
    if r.get("intent") in mandatory_escalation_intents and not r.get("escalate"):
        errors.append(f"Row {idx+1} [{tid}] has mandatory escalation intent '{r.get('intent')}' but escalate=False")

    # Check notes field is present and non-empty
    if not r.get("notes") or not str(r.get("notes")).strip():
        errors.append(f"Row {idx+1} [{tid}] missing annotator notes")

    # Check source text fidelity
    sub = subsample.get(tid)
    if not sub:
        errors.append(f"Row {idx+1} [{tid}] not found in subsample!")
    else:
        if r.get("customer_text") != sub["customer_text"]:
            errors.append(f"Row {idx+1} [{tid}] customer_text differs from subsample source")
        if r.get("brand_text") != sub["brand_text"]:
            errors.append(f"Row {idx+1} [{tid}] brand_text differs from subsample source")

# 5. Check gold_unlabeled_200.jsonl is genuinely unlabeled
with open("data/gold/gold_unlabeled_200.jsonl", "r", encoding="utf-8") as f:
    unlabeled = [json.loads(line) for line in f]

assert len(unlabeled) == 200, f"Expected 200 unlabeled records, got {len(unlabeled)}"
for idx, u in enumerate(unlabeled):
    for fld in ["intent", "escalate", "escalate_reason", "notes"]:
        if u.get(fld) is not None:
            errors.append(f"Unlabeled file row {idx+1} has non-null {fld}: {u.get(fld)}")

if errors:
    print(f"Validation failed with {len(errors)} errors:")
    for e in errors[:15]:
        print("  ", e)
    raise SystemExit(1)
else:
    print("SUCCESS: All 200 rows strictly pass schema, codebook escalation, source fidelity, and unlabeled holdout validation!")
