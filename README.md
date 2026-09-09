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

### 4. Running Baselines and Evaluation

Run the evaluation harness directly on the golden evaluation set (`data/gold/gold_eval_200.jsonl`):

```bash
# 1. Run Trivial Baseline (Majority Intent + Static Canned Reply)
python -m src.eval.run_baseline --type trivial

# 2. Run Simple Baseline (TF-IDF Retrieval + Heuristic Escalation Rules)
python -m src.eval.run_baseline --type simple

# 3. Run Full Proposed Agent (Intent Classifier + Vector RAG + Calibrated Escalation)
python -m src.eval.run_eval --agent proposed

# 4. Run LLM-as-a-Judge on outputs & compute human agreement
python -m src.eval.run_judge --eval-file reports/eval_results_proposed.jsonl
```

All summary tables, confusion matrices, and metrics will be printed to stdout and saved in `reports/`.

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
