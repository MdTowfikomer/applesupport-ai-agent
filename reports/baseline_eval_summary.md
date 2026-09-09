# Task T5 Baseline Evaluation Summary

## 1. Executive Summary

This report documents the baseline performance established by non-learning dummy baselines on `data/gold/gold_eval_200.jsonl`.

| Baseline | Target Task | Accuracy | Macro-F1 | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority Intent (`other`)** | Intent (10 classes) | **16.00%** | **2.76%** | N/A | N/A | N/A |
| **Always Escalate (`True`)** | Escalation (Binary) | **51.50%** | N/A | **51.50%** | **100.00%** | **67.99%** |

## 2. Intent Classification Breakdown (Majority Intent)

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

## 3. Escalation Triage Breakdown (Always Escalate)

- **Escalation Accuracy**: `51.50%` (103/200)
- **Precision (escalate=True)**: `51.50%` (103/200)
- **Recall (escalate=True)**: `100.00%` (103/103)
- **F1-Score (escalate=True)**: `67.99%`

| Actual \ Predicted | Pred: Escalate (True) | Pred: Auto-Handle (False) | Total Actual |
|:---|:---:|:---:|:---:|
| **Actual: Escalate (True)** | **TP = 103** | **FN = 0** | 103 |
| **Actual: Auto-Handle (False)** | **FP = 97** | **TN = 0** | 97 |
| **Total Predicted** | 200 | 0 | 200 |
