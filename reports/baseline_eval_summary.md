# AppleSupport Baseline Evaluation Summary (Tasks T5, T6 & T7)

## 1. Executive Summary & Benchmark Floor

This report documents empirical baseline performance established on `data/gold/gold_eval_200.jsonl`.

| Baseline Pipeline | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Escalation Precision | Escalation Recall | Escalation F1 | ROUGE-1 F1 | ROUGE-L F1 | BLEU-1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Simple Baseline Agent (T7)** | **55.00%** | **53.57%** | **59.50%** | **72.00%** | **34.95%** | **47.06%** | **28.56%** | **24.04%** | **24.13%** |
| **Trivial Baseline Agent (T6)** | 16.00% | 2.76% | 51.50% | 51.50% | 100.00% | 67.99% | 33.98% | 27.40% | 29.29% |
| *Majority-Intent Only (T5)* | 16.00% | 2.76% | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| *Always-Escalate Only (T5)* | N/A | N/A | 51.50% | 51.50% | 100.00% | 67.99% | N/A | N/A | N/A |

## 2. Simple Baseline Agent (Task T7)

- **Intent Classifier**: Keyword & regex pattern matcher adhering to codebook hierarchy (Rule 3: account > billing > hardware > symptoms > vague).
- **Escalation Triage**: Deterministic safety, legal, credential, billing, and screenshot-only trigger rules.
- **Resolution Retrieval**: TF-IDF cosine-similarity retriever indexed over **4800** historical non-holdout threads (strictly excluding all 200 holdouts).

### Intent Classification Breakdown

**Overall Accuracy**: `55.00%` (110/200)
**Macro-F1**: `53.57%` | **Weighted-F1**: `57.75%`

| Intent Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| `battery_power_issue` | 100.0% | 96.3% | 98.1% | 27 |
| `performance_crash_freeze` | 100.0% | 50.0% | 66.7% | 30 |
| `software_update_glitch` | 32.4% | 44.0% | 37.3% | 25 |
| `connectivity_network_issue` | 52.9% | 81.8% | 64.3% | 11 |
| `hardware_physical_accessory` | 33.3% | 58.3% | 42.4% | 12 |
| `account_access_security` | 71.4% | 75.0% | 73.2% | 20 |
| `billing_purchases_subscriptions` | 66.7% | 50.0% | 57.1% | 8 |
| `apps_feature_howto` | 80.0% | 38.7% | 52.2% | 31 |
| `vague_complaint_unclear` | 4.3% | 25.0% | 7.4% | 4 |
| `other` | 45.5% | 31.2% | 37.0% | 32 |

```text
Class Key:
   [1] = battery_power_issue
   [2] = performance_crash_freeze
   [3] = software_update_glitch
   [4] = connectivity_network_issue
   [5] = hardware_physical_accessory
   [6] = account_access_security
   [7] = billing_purchases_subscriptions
   [8] = apps_feature_howto
   [9] = vague_complaint_unclear
  [10] = other

Confusion Matrix (Rows: Actual / Columns: Predicted):
Actual \ Pred                [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]  [10]  Total
---------------------------------------------------------------------------------------------
 [1] battery_power_issue      26     0     0     0     0     0     0     0     1     0     27
 [2] performance_crash_freeze     0    15    11     1     2     0     0     0     1     0     30
 [3] software_update_glitch     0     0    11     4     5     2     0     1     2     0     25
 [4] connectivity_network_i..     0     0     0     9     0     1     0     1     0     0     11
 [5] hardware_physical_acce..     0     0     2     0     7     0     0     0     2     1     12
 [6] account_access_security     0     0     1     0     0    15     1     0     2     1     20
 [7] billing_purchases_subs..     0     0     0     0     0     2     4     0     1     1      8
 [8] apps_feature_howto        0     0     2     0     6     1     0    12     4     6     31
 [9] vague_complaint_unclear     0     0     0     0     0     0     0     0     1     3      4
[10] other                     0     0     7     3     1     0     1     1     9    10     32
---------------------------------------------------------------------------------------------
Total Predicted               26    15    34    17    21    21     6    15    23    22    200
```

### Escalation Triage Breakdown

- **Escalation Accuracy**: `59.50%` (119/200)
- **Precision (escalate=True)**: `72.00%` (36/50)
- **Recall (escalate=True)**: `34.95%` (36/103)
- **F1-Score (escalate=True)**: `47.06%`

| Actual \ Predicted | Pred: Escalate (True) | Pred: Auto-Handle (False) | Total Actual |
|:---|:---:|:---:|:---:|
| **Actual: Escalate (True)** | **TP = 36** | **FN = 67** | 103 |
| **Actual: Auto-Handle (False)** | **FP = 14** | **TN = 83** | 97 |
| **Total Predicted** | 50 | 150 | 200 |

### Reply Generation Lexical Overlap

- **ROUGE-1 F1**: `28.56%`
- **ROUGE-2 F1**: `12.57%`
- **ROUGE-L F1**: `24.04%`
- **BLEU-1**: `24.13%`
- **Mean Reply Length**: `25.3 words` (132.4 chars) vs Gold: `26.9 words` (140.3 chars)

## 3. Trivial Baseline Agent Configuration (Task T6)

- **Intent Decision**: Fixed majority class (`other`)
- **Escalation Routing**: Fixed majority decision (`escalate = True`, reason = `channel_transition`)
- **Canned Reply Template**: `"Thanks for reaching out to us. We'd like to help get this resolved. Please send us a DM so we can look into this with you: https://t.co/GDrqU22YpT"`

### Intent Classification Breakdown

**Overall Accuracy**: `16.00%` (32/200)
**Macro-F1**: `2.76%` | **Weighted-F1**: `4.41%`

| Intent Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| `battery_power_issue` | 0.0% | 0.0% | 0.0% | 27 |
| `performance_crash_freeze` | 0.0% | 0.0% | 0.0% | 30 |
| `software_update_glitch` | 0.0% | 0.0% | 0.0% | 25 |
| `connectivity_network_issue` | 0.0% | 0.0% | 0.0% | 11 |
| `hardware_physical_accessory` | 0.0% | 0.0% | 0.0% | 12 |
| `account_access_security` | 0.0% | 0.0% | 0.0% | 20 |
| `billing_purchases_subscriptions` | 0.0% | 0.0% | 0.0% | 8 |
| `apps_feature_howto` | 0.0% | 0.0% | 0.0% | 31 |
| `vague_complaint_unclear` | 0.0% | 0.0% | 0.0% | 4 |
| `other` | 16.0% | 100.0% | 27.6% | 32 |

```text
Class Key:
   [1] = battery_power_issue
   [2] = performance_crash_freeze
   [3] = software_update_glitch
   [4] = connectivity_network_issue
   [5] = hardware_physical_accessory
   [6] = account_access_security
   [7] = billing_purchases_subscriptions
   [8] = apps_feature_howto
   [9] = vague_complaint_unclear
  [10] = other

Confusion Matrix (Rows: Actual / Columns: Predicted):
Actual \ Pred                [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]  [10]  Total
---------------------------------------------------------------------------------------------
 [1] battery_power_issue       0     0     0     0     0     0     0     0     0    27     27
 [2] performance_crash_freeze     0     0     0     0     0     0     0     0     0    30     30
 [3] software_update_glitch     0     0     0     0     0     0     0     0     0    25     25
 [4] connectivity_network_i..     0     0     0     0     0     0     0     0     0    11     11
 [5] hardware_physical_acce..     0     0     0     0     0     0     0     0     0    12     12
 [6] account_access_security     0     0     0     0     0     0     0     0     0    20     20
 [7] billing_purchases_subs..     0     0     0     0     0     0     0     0     0     8      8
 [8] apps_feature_howto        0     0     0     0     0     0     0     0     0    31     31
 [9] vague_complaint_unclear     0     0     0     0     0     0     0     0     0     4      4
[10] other                     0     0     0     0     0     0     0     0     0    32     32
---------------------------------------------------------------------------------------------
Total Predicted                0     0     0     0     0     0     0     0     0   200    200
```

### Escalation Triage Breakdown

- **Escalation Accuracy**: `51.50%` (103/200)
- **Precision (escalate=True)**: `51.50%` (103/200)
- **Recall (escalate=True)**: `100.00%` (103/103)
- **F1-Score (escalate=True)**: `67.99%`

| Actual \ Predicted | Pred: Escalate (True) | Pred: Auto-Handle (False) | Total Actual |
|:---|:---:|:---:|:---:|
| **Actual: Escalate (True)** | **TP = 103** | **FN = 0** | 103 |
| **Actual: Auto-Handle (False)** | **FP = 97** | **TN = 0** | 97 |
| **Total Predicted** | 200 | 0 | 200 |

### Reply Generation Lexical Overlap

- **ROUGE-1 F1**: `33.98%`
- **ROUGE-2 F1**: `13.81%`
- **ROUGE-L F1**: `27.40%`
- **BLEU-1**: `29.29%`
- **Mean Reply Length**: `31.0 words` (146.0 chars) vs Gold: `26.9 words` (140.3 chars)
