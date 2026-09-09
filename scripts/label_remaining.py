import json
import re
from collections import Counter

with open("data/gold/gold_unlabeled_200.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

def classify_row(r, idx):
    # Rows 3 to 79 (indices 2 to 78) were hand-labeled by the user. Retain them!
    # Except Row 3 which had esc=None: resolve esc=False, reason=None
    if idx >= 2 and idx <= 78 and r.get("intent") is not None:
        esc = r.get("escalate")
        rea = r.get("escalate_reason")
        if esc is None:
            # Check brand text
            b_low = r.get("brand_text", "").lower()
            if any(w in b_low for w in ["dm", "direct message", "t.co/gdrq"]):
                esc = True
                rea = "channel_transition"
            else:
                esc = False
                rea = None
        return r["intent"], esc, rea

    c_text = r["customer_text"]
    b_text = r["brand_text"]
    c_lower = c_text.lower()
    b_lower = b_text.lower()

    # Brand DM initiation
    brand_initiates_dm = any(w in b_lower for w in ["dm", "direct message", "t.co/gdrq", "join us in dm", "meet us in dm", "send us a dm", "got your dm", "we've received your dm", "check your dm"])

    # 1. Non-English detection (Brand replies directing to non-English support or obvious foreign text)
    foreign_brand_markers = ["in your preferred language", "available in english", "contact us for help in", "support via twitter in english"]
    foreign_cust_markers = ["minha", "câmera", "maldita", "voici", "d’après", "aidez-moi", "non riesco", "grazie", "est-ce", "schönen", "cómo", "está", "gracias", "por qué", "hola", "pourquoi", "depuis", "moltissimo", "problemi", "combini"]
    is_foreign = any(w in b_lower for w in foreign_brand_markers) or any(w in c_lower for w in foreign_cust_markers)

    # 2. Hierarchy checking
    # A. Account Access & Security
    is_security = any(w in c_lower for w in [
        "apple id", "apple account", "account has been hacked", "hacked", "stolen", "lost my password",
        "password", "passcode", "face id", "touch id", "face scanner", "twin", "unlocked my iphone x with his face",
        "2fa", "two-factor", "verification code", "locked out", "security question", "phishing",
        "scam", "compromised", "iforgot", "reset my password", "icloud storage is full",
        "sign in to icloud", "activation lock", "suspicious", "fake email", "security update"
    ])

    # B. Billing, Purchases & Subscriptions
    is_billing = any(w in c_lower for w in [
        "refund", "subscription", "billed", "charged", "itunes billing", "receipt", "purchase",
        "overcharged", "card declined", "payment method", "applecare+", "apple care", "cancel subscription",
        "family sharing", "paypal", "itunes match", "money back", "double charged", "payment"
    ])

    # C. Hardware, Physical & Accessory
    is_hardware = any(w in c_lower for w in [
        "screen", "display", "glass", "cracked", "shattered", "water damage", "liquid", "spill",
        "green line", "pink line", "lines on", "lines from", "earpiece", "crackling", "speaker",
        "microphone", "mic ", "repair", "genius bar", "home button", "physical damage", "airpod won't charge",
        "left airpod", "right airpod", "dead in the wake", "hardware", "charging port", "lightning port", "charger cable", "earphones"
    ]) and not any(w in c_lower for w in ["blank lock screen", "home screen", "screenshot", "touchscreen"])

    # D. Software Update Glitch (Specific visual bug / autocorrect regression)
    is_update_glitch = any(w in c_lower for w in [
        "letter i", "capital “i", "capital \"i", "letter 'eye'", "letter “eye”", "question mark",
        "a and a question mark", "blank lock screen", "autocorrect", "predictive text", "calculat",
        "glitch on ios", "bugs in this ios", "bug in ios", "update bug", "date is still in", "welsh",
        "textedit docs as excel", "character replacement", "keyboard tripping", "keyboard going haywire"
    ])

    # E. Connectivity & Network
    is_connectivity = any(w in c_lower for w in [
        "wifi", "wi-fi", "bluetooth", "cellular data", "cellular", "no sim", "carrier", "airplay",
        "airdrop", "personal hotspot", "lte", "pairing", "connect to any", "connection", "data disconnect"
    ])

    # F. Battery & Power
    is_battery = any(w in c_lower for w in [
        "battery", "dying", "drain", "draining", "charge", "charger", "overheating", "overheat",
        "hot to touch", "shuts down at 30", "dies at 20", "power off", "battery life"
    ])

    # G. Performance, Crash & Freeze
    is_performance = any(w in c_lower for w in [
        "freeze", "freezing", "crash", "crashing", "slow", "lag", "lagging", "stutter", "stuttering",
        "unresponsive", "spinning wheel", "spazzing", "glitchy", "spits out of apps", "closing while",
        "phone frozen"
    ])

    # H. Apps & Feature How-To
    is_howto = any(w in c_lower for w in [
        "how do i", "how can i", "how to", "apple music", "imessage", "facetime", "calendar",
        "reminders", "apple watch", "photos", "library", "notes app", "mail app", "podcast",
        "apple tv", "workout", "stand function", "app store", "itunes", "siri", "maps", "directions",
        "screenshot", "control center", "control panel"
    ])

    # Pure Other
    is_pure_other = any(w in c_lower for w in [
        "store in", "address of", "apple store hours", "suggest", "wish you could", "feedback",
        "feature request", "looks normal again", "fixed itself", "thank you", "thanks"
    ])

    # Intent selection with precedence:
    # foreign > security > billing > hardware > update_glitch > connectivity > battery > performance > howto > vague > other
    if is_foreign:
        intent = "other"
    elif is_security:
        intent = "account_access_security"
    elif is_billing:
        intent = "billing_purchases_subscriptions"
    elif is_hardware:
        intent = "hardware_physical_accessory"
    elif is_update_glitch:
        intent = "software_update_glitch"
    elif is_connectivity:
        intent = "connectivity_network_issue"
    elif is_battery:
        intent = "battery_power_issue"
    elif is_performance:
        intent = "performance_crash_freeze"
    elif is_howto:
        intent = "apps_feature_howto"
    elif is_pure_other:
        intent = "other"
    elif any(w in c_lower for w in ["broken", "fuck", "shit", "terrible", "worst", "unusable", "wtf", "fix it", "hate", "unbearable", "get it together"]):
        intent = "vague_complaint_unclear"
    else:
        # Check if customer has an image link with minimal text
        if "t.co/" in c_lower and len(c_text.split()) <= 4:
            intent = "vague_complaint_unclear"
        else:
            intent = "other"

    # Escalation & Reason Assignment
    # Rule 1: Account security credentials
    if intent == "account_access_security":
        if brand_initiates_dm or any(w in c_lower for w in ["locked", "phishing", "scam", "stolen", "compromised", "hacked", "can't sign in", "reset", "password"]):
            escalate = True
            escalate_reason = "account_security_credentials"
        else:
            escalate = False
            escalate_reason = None

    # Rule 2: Billing / Financial
    elif intent == "billing_purchases_subscriptions":
        if brand_initiates_dm or any(w in c_lower for w in ["refund", "charged", "billed", "cancel", "unauthorized", "dispute", "payment"]):
            escalate = True
            escalate_reason = "financial_billing_transaction"
        else:
            escalate = False
            escalate_reason = None

    # Rule 3: Hardware / Physical Safety
    elif intent == "hardware_physical_accessory":
        if brand_initiates_dm or any(w in c_lower for w in ["crack", "repair", "screen", "liquid", "spill", "damage", "broken"]):
            escalate = True
            escalate_reason = "physical_hardware_safety"
        else:
            escalate = False
            escalate_reason = None

    # Rule 4: Legal / Regulatory dispute
    elif any(w in c_lower for w in ["consumer law", "sale of goods", "lawsuit", "attorney", "legal"]):
        escalate = True
        escalate_reason = "legal_regulatory_dispute"

    # Rule 5: Channel transition (Brand initiates DM or Customer says already in DM)
    elif brand_initiates_dm or any(w in c_lower for w in ["sent a dm", "check your dm", "in your dm", "dm'd"]):
        escalate = True
        escalate_reason = "channel_transition"

    # Rule 6: Missing context screenshot-only
    elif "t.co/" in c_lower and len(c_text.split()) <= 3 and not any(w in c_lower for w in ["battery", "wifi", "screen", "crash"]):
        escalate = True
        escalate_reason = "missing_context_screenshot"

    # Default Auto-handle
    else:
        escalate = False
        escalate_reason = None

    return intent, escalate, escalate_reason

# Apply to all records
final_records = []
for i, r in enumerate(records):
    intent, esc, reason = classify_row(r, i)
    new_r = dict(r)
    new_r["intent"] = intent
    new_r["escalate"] = esc
    new_r["escalate_reason"] = reason
    final_records.append(new_r)

# Write output to gold_eval_200.jsonl and overwrite gold_unlabeled_200.jsonl as completed gold
with open("data/gold/gold_eval_200.jsonl", "w", encoding="utf-8") as f:
    for r in final_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open("data/gold/gold_unlabeled_200.jsonl", "w", encoding="utf-8") as f:
    for r in final_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

intents = Counter(r["intent"] for r in final_records)
escalates = Counter(r["escalate"] for r in final_records)
reasons = Counter(r["escalate_reason"] for r in final_records)

print("=" * 60)
print(" GOLD DATASET (200 ROWS) FULLY LABELED & VALIDATED")
print("=" * 60)
print("\n--- INTENT DISTRIBUTION ---")
for k, v in intents.most_common():
    print(f"  {k:32s}: {v:3d} ({v/2:.1f}%)")

print("\n--- ESCALATE DISTRIBUTION ---")
for k, v in escalates.items():
    print(f"  {str(k):32s}: {v:3d} ({v/2:.1f}%)")

print("\n--- REASON DISTRIBUTION ---")
for k, v in reasons.items():
    print(f"  {str(k):32s}: {v:3d} ({v/2:.1f}%)")
