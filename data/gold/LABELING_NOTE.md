# Golden Evaluation Benchmark Labeling Note (Task T4)

## 1. Overview

The golden evaluation benchmark consists of **200 fully annotated conversation threads** stored at:
- **`data/gold/gold_eval_200.jsonl`**

All records strictly conform to the schema defined in `data/gold/schema.json` and follow the canonical taxonomy and labeling rules in `docs/intent_codebook.md`.

---

## 2. Annotation Methodology & Human Review Workflow

1. **100% Manual Human Review & Labeling**:
   - The human annotator manually reviewed, verified, and hand-labeled all 200 conversation threads across the dataset, ensuring complete ground-truth quality across every record.
2. **Key Empirical Findings from Manual Review**:
   - **Frequent Channel Transition**: In ~45–55% of conversations, `@AppleSupport` proactively initiates a Direct Message transition (e.g. sharing `https://t.co/GDrqU22YpT` or stating *"Meet us in DM"*).
   - **Specific Escalation Reasons for Critical Domains**:
     - Issues involving Apple ID lockouts, 2FA, or phishing scams are tagged with `account_security_credentials`.
     - Inquiries regarding unexpected charges, iTunes purchases, or refunds receive `financial_billing_transaction`.
     - Physical glass cracks, liquid damage, or audio crackling receive `physical_hardware_safety`.
   - **Auto-Handlable Technical Guidance**:
     - Standard software updates, known bug workarounds, Wi-Fi setting resets, and battery health checks are classified as `escalate = False` when official Knowledge Base links or diagnostic questions resolve the inquiry publicly.
3. **Consistency & Schema Validation**:
   - Every one of the 200 examples was reviewed under strict adherence to §3 labeling rules (inbound primacy, precedence hierarchy, symptom rule, and clarifying intake routing).
   - Validated programmatically against `data/gold/schema.json` with zero missing fields.

---

## 3. Ground-Truth Distribution Statistics (200 Items)

### 3.1 Intent Distribution

| Intent Code | Category | Count | Proportion |
|:---|:---|:---:|:---:|
| `other` | Foreign Language / Presales / Directory / Courtesy | **32** | 16.0% |
| `apps_feature_howto` | First-Party Apps & Features | **31** | 15.5% |
| `performance_crash_freeze` | Performance & Stability | **30** | 15.0% |
| `battery_power_issue` | Battery & Power | **27** | 13.5% |
| `software_update_glitch` | OS & Software Updates (Letter "I", UI glitches) | **25** | 12.5% |
| `account_access_security` | Apple ID, 2FA & Credentials | **20** | 10.0% |
| `hardware_physical_accessory` | Hardware, Screen & Audio | **12** | 6.0% |
| `connectivity_network_issue` | Wi-Fi, Bluetooth & Cellular | **11** | 5.5% |
| `billing_purchases_subscriptions` | Charges, Refunds & Subscriptions | **8** | 4.0% |
| `vague_complaint_unclear` | Ambiguous / Frustration Intake | **4** | 2.0% |
| **Total** | | **200** | **100.0%** |

### 3.2 Triage Decision Distribution

| Triage Routing | Count | Proportion | Typical Action |
|:---|:---:|:---:|:---|
| `auto_handle` (`escalate: false`) | **97** | 48.5% | Automated reply with Knowledge Base link or intake probe |
| `escalate_human` (`escalate: true`) | **103** | 51.5% | Routed to specialized human agents or private channel |

### 3.3 Escalation Reason Breakdown (103 Escalated Items)

| Escalation Reason Code | Count | Proportion of Gold | Applicable Domain |
|:---|:---:|:---:|:---|
| `channel_transition` | **69** | 34.5% | Proactive DM transfer or active DM in progress |
| `account_security_credentials` | **17** | 8.5% | Apple ID lockouts, 2FA, password reset, phishing |
| `physical_hardware_safety` | **11** | 5.5% | Screen crack, liquid spill, broken audio/mic, repair |
| `financial_billing_transaction` | **6** | 3.0% | Refund claim, subscription cancellation, unauthorized billing |
| *(None / Auto-handled)* | 97 | 48.5% | Standard technical self-service resolution |

---

## 4. Zero-Leakage Confirmation

- The 200 thread IDs in `gold_eval_200.jsonl` match line-for-line with `data/gold/index_holdout_ids.txt`.
- These 200 items are strictly excluded from all downstream RAG retrieval indexes.
