# Golden Evaluation Benchmark Sampling Documentation (T3)

## 1. Overview & Objective

- **Total Subsample Corpus Size**: `5,000` threads
- **Golden Benchmark Size**: `200` items (within the mandated 150–250 range)
- **Annotation State**: **Unlabeled** (`intent: null`, `escalate: null`, `escalate_reason: null`).
  *The agent must not invent gold labels. All ground truth will be annotated by the human in Task T4.*

## 2. Multi-Dimensional Stratification Strategy

Because `twcs.csv` is not strictly chronological, sampling partitions the file into three contiguous position blocks to ensure uniform representation across the entire file rather than clustering at the head:

### A. File Position Stratum (Start / Middle / End)

| Stratum Region | File Index Range | Target Allocation | Actual Sampled Items | Proportion |
|:---|:---:|:---:|:---:|:---:|
| `start` | 1–1,666 | ~67 | **67** | 33.5% |
| `middle` | 1,667–3,333 | ~67 | **67** | 33.5% |
| `end` | 3,334–5,000 | ~67 | **66** | 33.0% |

### B. Conversation Depth & Complexity Stratum

Customer service automation must handle both rapid, single-response inquiries and complex, multi-turn troubleshooting. The sample intentionally balances both:

| Conversation Depth | Turns | Actual Sampled Items | Proportion |
|:---|:---:|:---:|:---:|
| `2 turns` | 2 | **119** | 59.5% |
| `3 turns` | 3 | **19** | 9.5% |
| `4 turns` | 4 | **57** | 28.5% |
| `5 turns` | 5 | **2** | 1.0% |
| `6 turns` | 6 | **1** | 0.5% |
| `7 turns` | 7 | **2** | 1.0% |

- **2-Turn (Single exchange)**: `119` (59.5%)
- **3+ Turn (Multi-turn dialogue)**: `81` (40.5%)

### C. Weak Candidate Domain Balancing (Sampling-Time Only)

> [!NOTE]
> Weak candidate buckets were used solely during round-robin sampling to guarantee that low-frequency critical intents (such as billing disputes, Apple ID lockouts, and hardware damage) were not starved out by high-volume OS update rants. **These candidate buckets are NOT written as labels**.

| Candidate Weak Domain | Sampled Candidate Items | Proportion |
|:---|:---:|:---:|
| `battery_power_candidate` | 24 | 12.0% |
| `performance_crash_candidate` | 23 | 11.5% |
| `apps_features_candidate` | 23 | 11.5% |
| `connectivity_candidate` | 23 | 11.5% |
| `general_unclear_candidate` | 22 | 11.0% |
| `hardware_physical_candidate` | 22 | 11.0% |
| `account_security_candidate` | 21 | 10.5% |
| `billing_subscription_candidate` | 21 | 10.5% |
| `software_update_candidate` | 21 | 10.5% |

---

## 3. Strict Retrieval Leakage Holdout Split Rule

To prevent data leakage during RAG vector retrieval and historical reply generation:
1. Exactly `200` thread IDs are isolated into `data/gold/index_holdout_ids.txt`.
2. **Strict Rule**: When building the vector database / BM25 index in Task T7/T8, all `200` holdout thread IDs **MUST BE EXCLUDED** from the retrieval index.
3. The retrieval corpus consists strictly of the remaining `4,800` threads (`data/subsample/applesupport_threads_5k.jsonl` minus `index_holdout_ids.txt`).
4. An automated assertion will verify in the eval harness that no retrieved reference originates from the golden evaluation set.

---

## 4. Artifact Verification Checklist

- [x] `data/gold/schema.json` defined with valid types, enums, and required fields.
- [x] `data/gold/gold_unlabeled_200.jsonl` generated with exactly `200` items.
- [x] `data/gold/index_holdout_ids.txt` populated with `200` unique IDs.
- [x] All items have `intent: null`, `escalate: null`, `escalate_reason: null`, `notes: null`.
- [x] Ready for human labeling in Task T4.
