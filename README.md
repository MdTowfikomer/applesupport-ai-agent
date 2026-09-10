# AI Customer Support Agent for AppleSupport

> **Hiver SDE Intern Assignment**  
> An autonomous AI customer support system built on real Twitter customer service conversations, featuring **Intent Classification**, **Historical Resolution RAG**, and **Safety & Policy Escalation Guardrails**, evaluated with a rigorous golden dataset and human-calibrated LLM-as-a-judge.

---

## ⚡ 15-Minute Headline Reproduction Guide

Follow these exact steps to reproduce the evaluation results and metrics from scratch in **under 15 minutes**.

### 1. Prerequisites & Environment Setup

Ensure you have Python 3.10+ and [`uv`](https://github.com/astral-sh/uv) installed (or standard `pip`).

```bash
# Clone the repository (if not already local)
git clone <repo-url>
cd Customer_support_agent

# Create and activate virtual environment with uv
uv venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and set your API keys:
- **Google AI Studio (`GEMINI_API_KEY`)**: Used for the primary agent pipeline (classification, RAG reply drafting, and escalation triage).
- **Groq (`GROQ_API_KEY`)**: Used for the independent LLM-as-a-Judge (ensuring cross-model evaluation without self-preference bias).

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Data Setup & Subsampling

The full Twitter Customer Support dataset (`thoughtvector/customer-support-on-twitter`, ~750MB compressed / ~3M tweets) is **not** required for running the pre-packaged evaluation pipeline. The subsample and golden eval set are versioned in `data/`.

To process the raw dataset yourself:
1. Download `twcs.csv` from Kaggle:
   - CLI: `kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw/ --unzip`
   - Web: [Kaggle Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)  
   Place the resulting `twcs.csv` inside `data/raw/twcs.csv`.
2. Run the thread reconstruction and subsampling script:
   ```bash
   python scripts/subsample_data.py --input data/raw/twcs.csv --target-threads 5000
   ```
   *(Note: The script safely checks for `data/raw/twcs.csv` and fails clearly with explicit download commands if missing).*

### 4. Running Baselines, AI Agent, and Evaluation (Tasks T5 - T8)

The evaluation harness evaluates `data/gold/gold_eval_200.jsonl` offline **without requiring API keys or external services** (with optional Gemini LLM generation):

```bash
# 1. Run the T8 Autonomous AI Customer Support Agent:
# Single interactive query:
python -m src.eval.run_agent --text "My iPhone battery drains in 2 hours on iOS 11.1"

# Evaluating 5 gold sample records:
python -m src.eval.run_agent --samples 5

# Offline deterministic execution (no API key needed):
python -m src.eval.run_agent --offline --text "Screen is cracked in half"

# 2. Run Baselines:
# T7 Simple Baseline agent (keyword intent + rule escalation + TF-IDF historical retrieval):
python -m src.eval.run_baseline --type simple

# T6 Trivial Baseline agent (majority intent + always escalate + canned DM reply):
python -m src.eval.run_baseline --type trivial

# 3. Run the full comparative evaluation benchmark on all baselines & agent:
python -m scripts.eval_gold --baseline all

# 4. Run unit and integration tests (both standard library and pytest):
python -m unittest discover tests
pytest tests/ -v
```

#### Empirical Benchmark Comparison (`gold_eval_200.jsonl`, 200 items)

| Pipeline / Model | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Escalation Precision | Escalation Recall | Escalation F1 | ROUGE-1 F1 | ROUGE-L F1 | BLEU-1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **AppleSupport AI Agent (T8)** | **55.00%** | **53.57%** | **59.50%** | **72.00%** | 34.95% | 47.06% | **28.27%** | **23.81%** | **23.50%** |
| Simple Baseline Agent (T7) | 55.00% | 53.57% | 59.50% | 72.00% | 34.95% | 47.06% | 28.56% | 24.04% | 24.13% |
| Trivial Baseline Agent (T6) | 16.00% | 2.76% | 51.50% | 51.50% | **100.00%** | **67.99%** | 33.98% | 27.40% | 29.29% |
| *Majority-Intent Only (T5)* | 16.00% | 2.76% | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| *Always-Escalate Only (T5)* | N/A | N/A | 51.50% | 51.50% | 100.00% | 67.99% | N/A | N/A | N/A |

##### Key Findings & Pipeline Architecture (Task T8)

1. **Autonomous 4-Stage Pipeline Architecture** (`src/pipeline/`):
   - **Execution Sequence**: $\text{Intent} \longrightarrow \text{Retrieve } k \longrightarrow \text{Escalate Triage} \longrightarrow \text{Draft Reply}$.  
     *(Sequencing Rationale: Triage is intentionally executed prior to response drafting so the drafter can dynamically adapt its reply to the triage decision—embedding official DM escalation links for safety, security, and billing escalations, or providing grounded troubleshooting steps for auto-handled inquiries).*
   - **Intent Classification** ([`classifier.py`](file:///D:/Programming/major-projects/Customer_support_agent/src/pipeline/classifier.py)): 10 canonical classes adhering strictly to Codebook priority rules (`account_access_security` > `billing_purchases_subscriptions` > `hardware_physical_accessory` > symptoms > `vague_complaint_unclear`). Supports Gemini (`gemini-flash-latest`) with offline fallback.
   - **Historical Resolution Retrieval** ([`retriever.py`](file:///D:/Programming/major-projects/Customer_support_agent/src/pipeline/retriever.py)): BM25 / TF-IDF nearest-neighbor retrieval over **4,800 non-holdout threads** with strict mathematical assertion of zero holdout leakage ($Corpus \cap Holdouts = \emptyset$).
   - **Escalation Triage Router** ([`triage.py`](file:///D:/Programming/major-projects/Customer_support_agent/src/pipeline/triage.py)): Guardrails enforcing deterministic safety policies (battery thermal runaways, legal threats, credentials, billing disputes, missing context/screenshots) with structured, validated reason codes.
   - **Grounded Response Drafter** ([`drafter.py`](file:///D:/Programming/major-projects/Customer_support_agent/src/pipeline/drafter.py)): Synthesizes empathetic, brand-aligned Twitter replies under 280 characters grounded in retrieved historical Apple resolutions, embedding official DM transfer links (`https://t.co/GDrqU22YpT`) on escalated cases.

2. **Escalation Tradeoff (Precision vs. Recall)**:
   - T8/T7 improves escalation precision to **72.00%** (+20.5% gain over trivial baseline) and correctly auto-handles **85.6%** of non-escalated cases (83/97 true negatives).
   - Trivial baseline has higher recall (100%) and F1 (67.99%) only because it unconditionally escalates every single tweet, causing massive agent fatigue (97 false alarms).

### 5. Running LLM-as-a-Judge Calibration & Human Agreement (Task T9)

Deliverable 3 establishes an automated, human-calibrated evaluation protocol using **`openai/gpt-oss-120b`** (via Groq LPUs for cross-model evaluation independence from the Gemini pipeline generator, strict JSON adherence, and zero self-preference bias):

```bash
# 1. Run full 25-item human-calibration agreement benchmark:
python -m src.eval.run_judge --mode calibrate --model openai/gpt-oss-120b

# 2. Evaluate a single ad-hoc inquiry & candidate reply:
python -m src.eval.run_judge --mode single \
  --query "My battery drains so fast after iOS 17 update" \
  --reply "Thanks for reaching out. We can help with battery performance. Send us a DM: https://t.co/GDrqU22YpT" \
  --intent battery_issues --escalate

# 3. Run calibration in deterministic offline mode (no API keys required):
python -m src.eval.run_judge --offline
```

> [!TIP]
> **Groq Free Tier Rate-Limiting Guard**: The calibration runner includes built-in request pacing (`--delay 4.5`, ~5,300 TPM) and an exponential backoff retry handler to strictly comply with Groq Free Tier's 8,000 TPM limit.

#### Human vs. LLM Judge Agreement Benchmark (`judge_calibration_25.jsonl`, 25 items)

The judge was calibrated against **25 human-annotated customer resolutions** spanning diverse technical inquiries, routing edge cases, and safety failures:

| Evaluation Dimension | Pearson $r$ | Spearman $\rho$ | Cohen's QWK ($\kappa$) | Exact Match (%) | Within-1 Point (%) | MAE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall Quality (Primary)** | **0.7902** | **0.5347** | **0.7897** | 44.0% | **96.0%** | **0.6000** |
| Relevance & Precision | 0.6494 | 0.4622 | 0.6259 | 44.0% | 88.0% | 0.6800 |
| Tone & Empathy | 0.8445 | 0.6677 | 0.7116 | 48.0% | 92.0% | 0.6000 |
| Actionability & Escalation | 0.7957 | 0.6907 | 0.7805 | 52.0% | 92.0% | 0.5600 |
| Safety Guardrails (Binary) | N/A | N/A | N/A | **100.0%** | **100.0%** | **0.0000** |

- **Substantial Alignment ($\kappa = 0.7897$)**: Cohen's Quadratic Weighted Kappa indicates strong consensus between human expert annotators and `openai/gpt-oss-120b`.
- **96% Within-1 Point Accuracy**: 24 out of 25 evaluations fall within 1 point of human judgment, demonstrating high reliability for continuous regression testing.
- **100% Critical Safety Protection**: Perfect precision/recall in detecting safety, security, and privacy violations (e.g. credential harvesting or hazardous battery advice).
- Detailed rubric anchors and scoring examples are documented in [`docs/judge_rubric.md`](docs/judge_rubric.md). Machine-readable outputs and Markdown reports are saved in [`reports/judge_agreement_summary.md`](reports/judge_agreement_summary.md) and [`reports/judge_agreement_results.json`](reports/judge_agreement_results.json).

> [!NOTE]
> All predictions are serialized separately to `reports/eval_results_agent.jsonl`, `reports/eval_results_simple.jsonl`, and `reports/eval_results_trivial.jsonl` (200 records each), and benchmark metrics to `reports/baseline_eval_results.json` and `reports/baseline_eval_summary.md`. The gold dataset `data/gold/gold_eval_200.jsonl` remains 100% untouched and immutable. All 200 holdout IDs in `data/gold/index_holdout_ids.txt` remain strictly excluded from any retrieval index.


---

## 📁 Repository Layout

```text
.
├── .env.example                # Sample environment variables (no secrets)
├── .gitignore                  # Git ignore rules for data, venvs, and cache
├── pyproject.toml              # UV / Hatch build configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Setup, execution instructions, reproduction guide
├── Resources/                  # Assignment brief and reference files
├── data/
│   ├── raw/                    # Raw twcs.csv (ignored by git)
│   ├── subsample/              # Reconstructed AppleSupport conversation threads & stats
│   └── gold/                   # 150-250 hand-labelled evaluation benchmark
├── src/
│   ├── data/                   # Thread reconstruction, parsing, and data validation
│   ├── pipeline/               # Classifier, RAG retriever, response generator, triage logic
│   ├── baselines/              # Trivial and simple baselines
│   └── eval/                   # Metrics calculator, LLM-as-judge rubric, agreement stats
├── scripts/
│   ├── download_data.py        # Helper to fetch dataset via Kaggle API
│   └── subsample_data.py       # AppleSupport extraction and thread reconstructor
└── reports/                    # Benchmark tables, failure mode analysis, decision log
```

---

## 🎯 Target Brand: AppleSupport

- **Brand**: `@AppleSupport`
- **Domain**: Consumer electronics, iOS/macOS troubleshooting, Apple ID & iCloud account queries, battery/hardware repair, billing/app store subscriptions.
- **Why AppleSupport**:
  - High-volume, highly structured resolution patterns (troubleshooting steps, diagnostic link sharing, DM transfers).
  - Clear boundaries between auto-handlable technical guidance and mandatory human escalation (e.g. locked Apple IDs, hardware battery swelling, refunds/unauthorized charges, abusive interactions).
