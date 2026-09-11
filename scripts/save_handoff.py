"""
Script to save handoff context document to user's OS temporary directory (Task T12 -> T13).
"""

import os
import tempfile
from pathlib import Path

def main():
    temp_dir = Path(tempfile.gettempdir())
    handoff_path = temp_dir / "handoff_customer_support_agent_t12_to_t13.md"

    doc = """# Handoff Context Document: Customer Support AI Agent

## 1. Project Identity & Workspace
- **Corpus / Workspace**: `D:/Programming/major-projects/Customer_support_agent`
- **Target Domain**: `@AppleSupport` customer care on Twitter (`thoughtvector/customer-support-on-twitter`)
- **Git HEAD State**: Clean on branch `main` at commit `034c712` (all changes committed and verified).

---

## 2. Completed Milestones & Evaluated Deliverables

| Task / Deliverable | Status | Core Artifact / Path | Key Metrics / Grounding |
|:---|:---:|:---|:---|
| **T1–T3: Data Processing & Subsampling** | Completed | `data/subsample/apple_threads_5000.jsonl` | 5,000 reconstructed threads, zero holdout leakage |
| **T4: Golden Evaluation Benchmark** | Completed | `data/gold/gold_eval_200.jsonl` | 200 hand-labelled holdout items; strict schema validation |
| **T5–T6: Baselines (Majority & Trivial)** | Completed | `reports/eval_results_trivial.jsonl` | Majority intent 16.0% acc; Always-escalate 51.5% acc, 67.99% F1 |
| **T7: Simple Baseline Agent** | Completed | `reports/eval_results_simple.jsonl` | Keyword intent 55.0% acc; Rule escalation 59.5% acc, 72.0% prec |
| **T8: AppleSupport AI Agent** | Completed | `src/pipeline/` (`reports/eval_results_agent.jsonl`) | 4-stage pipeline (Intent -> Retrieve -> Triage -> Draft) |
| **T9: LLM-as-a-Judge Rubric & Calibration**| Completed | `reports/judge_agreement_summary.md` | `openai/gpt-oss-120b` via Groq: QWK kappa = 0.7640, 92.0% within-1 pt |
| **T10: Systematic Error Analysis & Dump** | Completed | `reports/error_dump_agent.jsonl` | 142 error cases dumped across 5 canonical failure modes |
| **T11: Failure Analysis & Headline Nuance** | Completed | `reports/failure_analysis_report.md` | 5 real Gold IDs analyzed; 4 misleading headline nuances |
| **T12: Comprehensive Final Report & Decision Log** | Completed | `reports/final_submission_report.md`, `reports/decision_log.md` | 12 non-obvious decisions; canonical escalation schema aligned |

---

## 3. Canonical Architecture & Validated Constraints

1. **Pipeline Sequencing Order**:
   - `Intent Classification` -> `BM25 Retrieval (k=3)` -> `Escalation Triage` -> `Grounded Draft Reply`.
   - *Rationale*: Triage must run before drafting so the response can conditionally embed an official DM link (`https://t.co/GDrqU22YpT`) or provide self-service troubleshooting.
2. **Deterministic Holdout Boundary**:
   - 200 holdout thread IDs (`data/gold/index_holdout_ids.txt`) are excluded from the 4,800-thread historical corpus ($Corpus \\cap Holdouts = \\emptyset$).
3. **Canonical 10-Class Intent Taxonomy**:
   - `battery_power_issue`, `performance_crash_freeze`, `software_update_glitch`, `connectivity_network_issue`, `hardware_physical_accessory`, `account_access_security`, `billing_purchases_subscriptions`, `apps_feature_howto`, `vague_complaint_unclear`, `other`.
4. **Canonical 6 Escalation Reason Codes**:
   - `account_security_credentials`, `financial_billing_transaction`, `physical_hardware_safety`, `legal_regulatory_dispute`, `channel_transition`, `missing_context_screenshot`.
5. **Safety Invariant**:
   - Battery thermal hazards (swollen, smoking, overheating) and credential harvesting (asking for 2FA, passwords) are deterministic hard-fail gates in the judge (`safety=False`, `overall_quality=1`).

---

## 4. Test Suite Status
- **Total Tests**: **44 passing** across `tests/test_agent_pipeline.py`, `tests/test_eval_harness.py`, `tests/test_judge.py`, and `tests/test_error_analyzer.py`.
- **Run Command**: `.venv\\Scripts\\pytest tests/ -v` or `python -m unittest discover tests`.

---

## 5. Next Session Focus: Pipeline Improvement vs. T13 Clean-Shell Dry-Run

The user raised two strategic questions:
1. **Should we improve the pipeline before T13?**
   - *Analysis*: All 5 assignment deliverables are fully implemented, verified, benchmarked, and committed. The assignment explicitly values proving what works, documenting failure modes, and operational honesty over endless optimization ("The proof is worth more than the system").
   - *Recommendation*:
     - **Path A (Proceed to T13 Clean-Shell Dry-Run)**: Verify the evaluator reproduction experience end-to-end (fresh venv, dependencies, offline execution, CLI commands).
     - **Path B (Targeted Pipeline Polish)**: If desiring higher metrics before T13, tackle the top failure modes documented in Deliverable 4:
       - Mode 1: Priority heuristic favoring third-party app crashes (`performance_crash_freeze`) over OS update context.
       - Mode 2: Invariant enforcement preventing drafter from outputting DM links if triage returned `escalate=False`.
       - Mode 3: Language pre-filter detecting Portuguese/Spanish and routing directly to official language deflection.
2. **T13 Clean-Shell Dry-Run Objectives**:
   - Follow `README.md` step-by-step in a simulated clean environment.
   - Verify:
     - `python -m src.eval.run_agent --offline --text "Screen is cracked"`
     - `python -m src.eval.run_baseline --type simple`
     - `python -m src.eval.run_baseline --type trivial`
     - `python -m scripts.eval_gold --baseline all`
     - `pytest tests/ -v`
     - Complete all checks in under 15 minutes.

---

## 6. Suggested Skills for Next Agent
- **antigravity-guide** (`C:\\Users\\MD Towfik Omer\\.gemini\\antigravity-cli\\builtin\\skills\\antigravity_guide\\SKILL.md`): For Antigravity environment management, slash commands, and clean session transitions.
- **agy-customizations** (`C:\\Users\\MD Towfik Omer\\.gemini\\antigravity-cli\\builtin\\skills\\agy-customizations\\SKILL.md`): For custom subagent or sidecar workflow automation.

---

*Handoff written to OS temporary directory without sensitive credentials. All evaluation numbers reconcile with repo reports.*
"""
    handoff_path.write_text(doc, encoding="utf-8")
    print(f"Handoff successfully written to: {handoff_path}")

if __name__ == "__main__":
    main()
