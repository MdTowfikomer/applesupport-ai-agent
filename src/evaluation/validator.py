"""
Dataset Validation Module for Golden Evaluation Set (T4/T5).
Enforces schema compliance, codebook integrity, and zero-leakage holdout isolation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

ALLOWED_INTENTS = [
    "battery_power_issue",
    "performance_crash_freeze",
    "software_update_glitch",
    "connectivity_network_issue",
    "hardware_physical_accessory",
    "account_access_security",
    "billing_purchases_subscriptions",
    "apps_feature_howto",
    "vague_complaint_unclear",
    "other"
]

ALLOWED_ESCALATE_REASONS = [
    "account_security_credentials",
    "financial_billing_transaction",
    "physical_hardware_safety",
    "legal_regulatory_dispute",
    "channel_transition",
    "missing_context_screenshot"
]

class GoldValidationError(Exception):
    """Raised when the golden evaluation dataset fails validation."""
    pass

def validate_gold_records(
    records: List[Dict[str, Any]],
    holdout_ids_path: Optional[Path] = None,
    expected_count: int = 200
) -> Dict[str, Any]:
    """
    Validates gold evaluation records against strict T4/T5 requirements:
    - Exactly 200 rows (or expected_count)
    - No duplicate thread IDs
    - No missing intent labels; all intents in canonical taxonomy
    - escalate is boolean
    - escalate_reason uses only allowed codebook codes or None
    - Reason consistency: escalate=True requires reason; escalate=False requires None/empty
    - Holdout isolation: thread IDs must align with index_holdout_ids.txt
    """
    errors: List[str] = []

    # 1. Exact record count check
    if len(records) != expected_count:
        errors.append(f"Expected exactly {expected_count} records, got {len(records)}")

    # 2. Duplicate ID check
    seen_ids = set()
    duplicates = []
    for idx, r in enumerate(records):
        tid = r.get("thread_id")
        if not tid:
            errors.append(f"Row {idx+1} missing 'thread_id'")
        elif tid in seen_ids:
            duplicates.append(tid)
        else:
            seen_ids.add(tid)

    if duplicates:
        errors.append(f"Found duplicate thread IDs: {sorted(list(set(duplicates)))}")

    # 3. Holdout IDs alignment
    if holdout_ids_path and Path(holdout_ids_path).exists():
        with open(holdout_ids_path, "r", encoding="utf-8") as f:
            holdout_ids = [line.strip() for line in f if line.strip()]
        
        if len(holdout_ids) != expected_count:
            errors.append(f"Holdout IDs file has {len(holdout_ids)} entries, expected {expected_count}")
        
        gold_ids = [r.get("thread_id") for r in records if r.get("thread_id")]
        if gold_ids != holdout_ids:
            mismatched = set(gold_ids).symmetric_difference(set(holdout_ids))
            errors.append(f"Gold IDs do not match holdout file line-for-line! Mismatched: {mismatched}")

    # 4. Field-level validation
    for idx, r in enumerate(records):
        tid = r.get("thread_id", f"row_{idx+1}")
        
        # Required core fields
        for field in ["thread_id", "customer_tweet_id", "customer_text", "brand_text"]:
            if field not in r or r[field] is None:
                errors.append(f"Row {idx+1} [{tid}] missing required field '{field}'")

        # Intent validation
        intent = r.get("intent")
        if intent is None or intent == "":
            errors.append(f"Row {idx+1} [{tid}] missing intent label")
        elif intent not in ALLOWED_INTENTS:
            errors.append(f"Row {idx+1} [{tid}] has invalid intent '{intent}'. Allowed: {ALLOWED_INTENTS}")

        # Escalate validation
        escalate = r.get("escalate")
        if not isinstance(escalate, bool):
            errors.append(f"Row {idx+1} [{tid}] 'escalate' must be boolean (got {type(escalate).__name__}: {escalate})")

        # Escalate reason validation
        reason = r.get("escalate_reason")
        if escalate is True:
            if not reason:
                errors.append(f"Row {idx+1} [{tid}] escalate=True requires an escalate_reason")
            elif reason not in ALLOWED_ESCALATE_REASONS:
                errors.append(f"Row {idx+1} [{tid}] invalid escalate_reason '{reason}'. Allowed: {ALLOWED_ESCALATE_REASONS}")
        elif escalate is False:
            if reason is not None and reason != "":
                errors.append(f"Row {idx+1} [{tid}] escalate=False requires null/empty escalate_reason (got '{reason}')")

    if errors:
        raise GoldValidationError(
            f"Gold validation failed with {len(errors)} errors:\n" + "\n".join(f"  - {e}" for e in errors[:20])
        )

    return {
        "status": "VALID",
        "total_records": len(records),
        "unique_threads": len(seen_ids),
        "intents_found": sorted(list({r["intent"] for r in records if r.get("intent")})),
    }

def load_and_validate_gold(
    gold_path: Path,
    holdout_ids_path: Optional[Path] = None,
    expected_count: int = 200
) -> List[Dict[str, Any]]:
    """Loads a gold JSONL file and strictly validates all constraints."""
    gold_path = Path(gold_path)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold evaluation file not found: {gold_path}")

    with open(gold_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    validate_gold_records(records, holdout_ids_path=holdout_ids_path, expected_count=expected_count)
    return records
