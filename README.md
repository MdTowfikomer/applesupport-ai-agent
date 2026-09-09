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

### 4. Running Baselines and Evaluation (Task T5)

The evaluation harness evaluates `data/gold/gold_eval_200.jsonl` offline **without requiring API keys or external services**.

```bash
# Run the evaluation harness on gold with sanity-check baselines:
python -m scripts.eval_gold

# Run unit and integration tests:
pytest tests/ -v
# or via standard library:
python -m unittest discover tests
```

#### Expected T5 Output & Baseline Benchmark Results

| Baseline | Evaluated Task | Accuracy | Macro-F1 | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority-Intent (`other`)** | Intent (10 classes) | **16.00%** | **2.76%** | N/A | N/A | N/A |
| **Always-Escalate (`True`)** | Escalation (Binary) | **51.50%** | N/A | **51.50%** | **100.00%** | **67.99%** |

> [!NOTE]
> These are non-learning, rule-free sanity-check baselines designed to establish the empirical performance floor on the 200-sample gold benchmark. All outputs are serialized to `reports/baseline_eval_results.json` without altering the gold JSONL dataset. Holdout IDs in `data/gold/index_holdout_ids.txt` remain strictly excluded from retrieval indexes.

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
