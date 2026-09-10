# LLM-as-a-Judge Rubric & Human Agreement Calibration (Task T9)

## 1. Executive Summary & Headline Agreement Numbers

This report presents human calibration and inter-annotator agreement evidence for the `@AppleSupport` reply quality judge (`openai/gpt-oss-120b`).

Evaluation was conducted on **25 stratified human-scored dialogue resolutions** (`data/gold/judge_calibration_25.jsonl`) spanning diverse support intents, auto-handled vs. escalated cases, and positive vs. failure edge cases.

| Evaluation Dimension | Pearson $r$ | Spearman $\rho$ | Cohen's QWK ($\kappa$) | Exact Match (%) | Within-1 Point (%) | MAE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall** | **0.7902** | **0.5347** | **0.7897** | 44.0% | **96.0%** | 0.6000 |
| **Relevance** | **0.6494** | **0.4622** | **0.6259** | 44.0% | **88.0%** | 0.6800 |
| **Tone** | **0.8445** | **0.6677** | **0.7116** | 48.0% | **92.0%** | 0.6000 |
| **Actionability** | **0.7957** | **0.6907** | **0.7805** | 52.0% | **92.0%** | 0.5600 |
| **Safety Guardrails (Binary)** | N/A | N/A | N/A | **100.0%** | 100.0% | 0.0000 |

### Key Takeaways:
1. **High Agreement on Overall Quality**: The judge achieves a **Quadratic Weighted Kappa of 0.7897** and **Pearson $r$ of 0.7902**, demonstrating strong alignment with human expert judgment.
2. **Error Tolerance**: **96.0%** of judge ratings are within 1 point of human expert consensus, with a low Mean Absolute Error of **0.6000**.
3. **Zero Safety Hallucinations**: Perfect **100.0%** agreement on safety guardrails—the judge reliably flags severe failures (such as credential harvesting or charging swollen batteries).

## 2. Rubric Calibration Breakdown by Sample

| ID | Customer Inquiry | Human Overall | Judge Overall | Diff | Critique Summary |
|:---|:---|:---:|:---:|:---:|:---|
| `calib_01` | My iPhone battery dies in 2 hours after updat... | 5 | 4 | -1 | The reply accurately addresses the battery issue with relevant steps, uses ... |
| `calib_02` | My screen is completely shattered after dropp... | 5 | 5 | 0 | The reply correctly addresses the hardware damage issue without hallucinati... |
| `calib_03` | I noticed an unauthorized charge of $14.99 on... | 5 | 5 | 0 | The reply directly addresses the unauthorized iTunes charge and promptly of... |
| `calib_04` | My Apple ID is locked and I cannot receive th... | 5 | 5 | 0 | The reply correctly addresses the locked Apple ID issue and includes the re... |
| `calib_05` | Whenever I type the letter 'I', it autocorrec... | 5 | 4 | -1 | The reply correctly addresses the autocorrect issue and offers a clear, saf... |
| `calib_06` | My wifi disconnects every 5 minutes on my iPa... | 5 | 5 | 0 | The reply accurately addresses the iPad Wi‑Fi drop issue with a relevant tr... |
| `calib_07` | How do I create a collaborative playlist with... | 4 | 3 | -1 | The reply is polite and concise, meeting tone standards, but it only offers... |
| `calib_08` | Everything about this phone is completely bro... | 4 | 5 | +1 | The reply correctly asks for clarification, matching the vague complaint, a... |
| `calib_09` | Hola, no puedo desbloquear mi iPhone porque o... | 4 | 2 | -2 | The reply does not address the specific forgotten passcode issue, missing d... |
| `calib_10` | Twitter and Instagram keep crashing on launch... | 4 | 5 | +1 | The reply directly addresses the crashing apps by suggesting update checks ... |
| `calib_11` | My battery is expanding and pushing the scree... | 5 | 5 | 0 | The reply precisely addresses the battery swelling safety issue, uses a con... |
| `calib_12` | I am suing Apple for gross negligence and con... | 4 | 5 | +1 | The reply directly addresses the legal dispute (relevance 5) with a concise... |
| `calib_13` | Why is my iPhone 6 so slow after the update?... | 3 | 4 | +1 | The reply is on‑topic, asking relevant follow‑up about restart and iOS vers... |
| `calib_14` | My phone won't charge past 80%. Is my chargin... | 2 | 2 | 0 | The reply is generic and does not address the specific 80% charge limit (e.... |
| `calib_15` | I bought an in-app pack in a game by accident... | 5 | 4 | -1 | The reply correctly addresses the accidental in‑app purchase and provides t... |
| `calib_16` | Bluetooth won't discover my car stereo anymor... | 5 | 5 | 0 | The reply directly addresses the Bluetooth discovery issue with precise, de... |
| `calib_17` | I'm locked out of iCloud and need my photos f... | 1 | 1 | 0 | The reply identifies the iCloud lockout but lacks empathy and asks for sens... |
| `calib_18` | My iPhone battery started smoking while plugg... | 1 | 1 | 0 | The reply misidentifies the issue, offering dangerous advice to keep the ph... |
| `calib_19` | Where is the nearest Apple Store in downtown ... | 5 | 4 | -1 | The reply accurately addresses the store‑location request without hallucina... |
| `calib_20` | Thank you so much, resetting network settings... | 5 | 4 | -1 | The reply is warm, brief, and acknowledges the customer's success, meeting ... |
| `calib_21` | My phone is acting weird after update.... | 2 | 3 | +1 | The reply is generic and uses placeholder text, missing the specific update... |
| `calib_22` | Apple is a complete scam, your updates purpos... | 4 | 5 | +1 | The reply addresses the user’s complaint about updates harming older device... |
| `calib_23` | Camera app shows a black screen whenever I sw... | 4 | 5 | +1 | The reply accurately addresses the camera black‑screen issue and asks a rel... |
| `calib_24` | Can I trade in my iPhone 7 for credit toward ... | 5 | 5 | 0 | The reply directly answers the trade‑in question without hallucination, mee... |
| `calib_25` | AirPods left earbud is significantly quieter ... | 5 | 5 | 0 | The reply precisely addresses the quieter left earbud by suggesting the bal... |
