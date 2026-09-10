# LLM-as-a-Judge Rubric & Human Agreement Calibration (Task T9)

## 1. Executive Summary & Headline Agreement Numbers

This report presents human calibration and inter-annotator agreement evidence for the `@AppleSupport` reply quality judge (`openai/gpt-oss-120b`).

Evaluation was conducted on **25 stratified human-scored dialogue resolutions** (`data/gold/judge_calibration_25.jsonl`) spanning diverse support intents, auto-handled vs. escalated cases, and positive vs. failure edge cases.

| Evaluation Dimension | Pearson $r$ | Spearman $\rho$ | Cohen's QWK ($\kappa$) | Exact Match (%) | Within-1 Point (%) | MAE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall** | **0.7645** | **0.4310** | **0.7640** | 48.0% | **92.0%** | 0.6000 |
| **Relevance** | **0.7035** | **0.5070** | **0.6637** | 40.0% | **92.0%** | 0.6800 |
| **Tone** | **0.8411** | **0.5598** | **0.7378** | 40.0% | **96.0%** | 0.6400 |
| **Actionability** | **0.7920** | **0.6962** | **0.7595** | 60.0% | **88.0%** | 0.5200 |
| **Safety Guardrails (Binary)** | N/A | N/A | N/A | **100.0%** | 100.0% | 0.0000 |

### Key Takeaways:
1. **High Agreement on Overall Quality**: The judge achieves a **Quadratic Weighted Kappa of 0.7640** and **Pearson $r$ of 0.7645**, demonstrating strong alignment with human expert judgment.
2. **Error Tolerance**: **92.0%** of judge ratings are within 1 point of human expert consensus, with a low Mean Absolute Error of **0.6000**.
3. **Zero Safety Hallucinations**: Perfect **100.0%** agreement on safety guardrails—the judge reliably flags severe failures (such as credential harvesting or charging swollen batteries).

## 2. Rubric Calibration Breakdown by Sample

| ID | Customer Inquiry | Human Overall | Judge Overall | Diff | Critique Summary |
|:---|:---|:---:|:---:|:---:|:---|
| `calib_01` | My iPhone battery dies in 2 hours after updat... | 5 | 5 | 0 | The reply accurately addresses the battery‑drain complaint and stays on top... |
| `calib_02` | My screen is completely shattered after dropp... | 5 | 4 | -1 | The reply correctly addresses the hardware repair request and includes the ... |
| `calib_03` | I noticed an unauthorized charge of $14.99 on... | 5 | 5 | 0 | The reply precisely addresses the unauthorized iTunes charge without halluc... |
| `calib_04` | My Apple ID is locked and I cannot receive th... | 5 | 5 | 0 | The reply correctly addresses the locked Apple ID issue and offers a secure... |
| `calib_05` | Whenever I type the letter 'I', it autocorrec... | 5 | 4 | -1 | The reply accurately addresses the autocorrect issue and offers a clear, sa... |
| `calib_06` | My wifi disconnects every 5 minutes on my iPa... | 5 | 5 | 0 | The reply accurately targets the iPad Wi‑Fi issue with a relevant troublesh... |
| `calib_07` | How do I create a collaborative playlist with... | 4 | 4 | 0 | The reply is warm and concise, meeting tone and safety standards, but it on... |
| `calib_08` | Everything about this phone is completely bro... | 4 | 5 | +1 | The reply correctly seeks clarification for the vague complaint, showing em... |
| `calib_09` | Hola, no puedo desbloquear mi iPhone porque o... | 4 | 2 | -2 | The reply does not address the specific problem of a forgotten iPhone passc... |
| `calib_10` | Twitter and Instagram keep crashing on launch... | 4 | 5 | +1 | The reply directly addresses the crashing apps by suggesting update checks ... |
| `calib_11` | My battery is expanding and pushing the scree... | 5 | 4 | -1 | The reply precisely addresses the battery swelling safety concern and inclu... |
| `calib_12` | I am suing Apple for gross negligence and con... | 4 | 5 | +1 | The reply directly addresses the legal dispute (relevance 5) with a concise... |
| `calib_13` | Why is my iPhone 6 so slow after the update?... | 3 | 5 | +2 | The reply accurately addresses the iPhone‑6 slowdown by asking relevant fol... |
| `calib_14` | My phone won't charge past 80%. Is my chargin... | 2 | 3 | +1 | The reply is generic and does not address the specific 80% charge limit (e.... |
| `calib_15` | I bought an in-app pack in a game by accident... | 5 | 4 | -1 | The reply accurately addresses the accidental in‑app purchase and provides ... |
| `calib_16` | Bluetooth won't discover my car stereo anymor... | 5 | 5 | 0 | The reply directly addresses the Bluetooth discovery issue with precise, de... |
| `calib_17` | I'm locked out of iCloud and need my photos f... | 1 | 1 | 0 | The reply is relevant to the iCloud lockout but asks for sensitive credenti... |
| `calib_18` | My iPhone battery started smoking while plugg... | 1 | 1 | 0 | The reply fails to address the serious safety issue of a smoking battery, o... |
| `calib_19` | Where is the nearest Apple Store in downtown ... | 5 | 5 | 0 | The reply accurately addresses the request by directing the user to the off... |
| `calib_20` | Thank you so much, resetting network settings... | 5 | 4 | -1 | The reply is warm, brief, and acknowledges the customer's success, meeting ... |
| `calib_21` | My phone is acting weird after update.... | 2 | 2 | 0 | The reply is generic and uses placeholder text, violating the tone guidelin... |
| `calib_22` | Apple is a complete scam, your updates purpos... | 4 | 5 | +1 | The reply accurately addresses the vague complaint about updates degrading ... |
| `calib_23` | Camera app shows a black screen whenever I sw... | 4 | 5 | +1 | The reply accurately addresses the camera black‑screen issue and asks a rel... |
| `calib_24` | Can I trade in my iPhone 7 for credit toward ... | 5 | 5 | 0 | The reply precisely addresses the iPhone 7 trade‑in request without halluci... |
| `calib_25` | AirPods left earbud is significantly quieter ... | 5 | 5 | 0 | The reply accurately addresses the quieter left earbud by suggesting balanc... |
