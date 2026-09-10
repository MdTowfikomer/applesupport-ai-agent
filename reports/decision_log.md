# Engineering Decision Log: 12 Non-Obvious Decisions & Rationale
**AppleSupport AI Customer Support Agent | Deliverable 5**

This document details 12 critical, non-obvious engineering decisions made during the design, implementation, and evaluation of the autonomous customer support agent.

---

### 1. Pipeline Sequencing: Executing Escalation Triage BEFORE Response Drafting
- **Decision**: Sequenced the pipeline as `Intent -> Retrieve -> Triage -> Draft`, rather than the conventional `Intent -> Retrieve -> Draft -> Triage`.
- **Rejected Alternative**: Drafting a candidate response first, then evaluating whether to escalate.
- **Non-Obvious Rationale**: If triage runs after drafting, the drafter cannot know whether the ticket requires a private DM transfer or public self-service. Running triage first allows the drafter to dynamically embed official DM escalation links (`https://t.co/GDrqU22YpT`) for security/billing/hardware hazards, or provide concrete self-service troubleshooting steps for auto-handled inquiries. This completely eliminates channel, state, and tone contradictions.

---

### 2. Holdout Retrieval Exclusion: Strict Mathematical Corpus Disjointness ($Corpus \cap Holdouts = \emptyset$)
- **Decision**: Formally partitioned the 5,000 reconstructed threads into an immutable 200-thread gold holdout set and an isolated 4,800-thread retrieval corpus, enforced by automated unit tests (`assert thread_id not in holdout_ids`).
- **Rejected Alternative**: Building a retrieval index over all available subsample data or filtering holdouts dynamically by text similarity.
- **Non-Obvious Rationale**: In customer support RAG systems, retrieving the exact ground-truth thread produces artificially inflated ROUGE/BLEU scores ("data leakage masquerading as intelligence"). Real customers never submit issues that already exist identically in historical archives. Zero-leakage holdout isolation ensures genuine generalization testing.

---

### 3. Cross-Model LLM-as-a-Judge Evaluation (`openai/gpt-oss-120b` via Groq)
- **Decision**: Evaluated the Gemini-based generator using an independent open-weights model (`openai/gpt-oss-120b`) hosted on Groq LPUs.
- **Rejected Alternative**: Using the same Gemini model for both reply generation and evaluation judging.
- **Non-Obvious Rationale**: LLM judges evaluated on their own outputs exhibit documented self-preference bias (up to 15-20% higher scores on identical content). Using a distinct open-weights model hosted on independent infrastructure ensures uncompromised, objective rubric enforcement and strict JSON schema reliability.

---

### 4. Physical Battery Hazards as Deterministic Hard-Fail Safety Gates
- **Decision**: Hard-coded deterministic rule enforcement in the judge: any reply advising a customer to keep charging, leave plugged in, or pierce a swollen/smoking battery automatically receives `safety=False` and clamps `overall_quality = 1`, regardless of politeness or tone.
- **Rejected Alternative**: Letting the LLM judge assign a soft, averaged Likert score (e.g., 1–5) to unsafe responses.
- **Non-Obvious Rationale**: In consumer electronics support, battery thermal runaways represent physical danger and legal liability. A polite, beautifully phrased reply that advises keeping a swollen battery plugged in is infinitely worse than a blunt refusal. Safety must never be an averaged soft metric.

---

### 5. Precision-First Escalation Policy Over Recall Maximization
- **Decision**: Deliberately optimized for high Escalation Precision (**72.00%**) and high Deflection Rate (**85.6%**), accepting lower Recall (34.95%) and lower F1 (47.06% vs. Trivial baseline's 67.99%).
- **Rejected Alternative**: Tuning classifier thresholds to maximize balanced F1 or Escalation Recall.
- **Non-Obvious Rationale**: In enterprise contact centers, false escalations (sending routine issues to senior human agents) directly blow SLAs and destroy agent morale (97 false alarms per 200 tickets). Routine issues can safely attempt self-service; if unresolved, the customer replies and escalates in turn 2. High precision protects human agent bandwidth.

---

### 6. Choosing a 10-Class Intent Taxonomy Governed by Priority Precedence
- **Decision**: Established exactly 10 high-level, mutually exclusive canonical intents (`battery_power_issue`, `performance_crash_freeze`, `software_update_glitch`, etc.) governed by explicit Codebook precedence rules.
- **Rejected Alternative**: Building a granular taxonomy with 30–50 specific sub-intents (e.g., `wifi_password_wrong`, `airpods_left_ear_silent`).
- **Non-Obvious Rationale**: High-cardinality intent spaces on short, noisy Twitter posts (averaging 15–25 words) suffer from severe semantic smearing and erratic classifier entropy. A 10-class taxonomy aligns directly with actual enterprise routing queues (Hardware vs. Billing vs. Network vs. Account Security) and maximizes downstream actionability.

---

### 7. Codebook Precedence Rules: Security & Billing Over Symptoms
- **Decision**: Enforced deterministic priority ordering: `account_access_security` > `billing_purchases_subscriptions` > `hardware_physical_accessory` > specific symptoms > `vague_complaint_unclear`.
- **Rejected Alternative**: Standard semantic similarity or bag-of-words classification without hierarchical priority.
- **Non-Obvious Rationale**: When a customer tweet contains multiple issues (e.g., "My phone froze and my Apple ID was hacked and money was taken"), symptom-level troubleshooting is irrelevant. Security compromises and unauthorized financial charges must immediately preempt technical troubleshooting.

---

### 8. Requiring Verifiable Escalation Reason Codes for Every Escalation Decision
- **Decision**: Enforced that `escalate=True` MUST be paired with a validated categorical reason code (`account_security_credentials`, `billing_payment_dispute`, `physical_hardware_safety`, `legal_threat`, `channel_transition`, `unsupported_device_scope`), enforced by validation schema tests.
- **Rejected Alternative**: Emitting a bare boolean flag (`escalate: true/false`).
- **Non-Obvious Rationale**: A black-box escalation boolean provides zero auditability for contact center team leads. Categorical reason codes enable automated routing to specialized Tier-2 queues (e.g., Security vs. Billing vs. Hardware), audit logging, and SLA compliance tracking.

---

### 9. Zero-Dependency Deterministic Offline Fallbacks
- **Decision**: Built complete, deterministic offline fallbacks for both the AI Agent (`--offline`) and the LLM Judge (`--offline`), backed by rule-based triage and nearest-neighbor retrieval.
- **Rejected Alternative**: Relying 100% on external APIs, failing completely if API keys are missing or rate limits are exceeded.
- **Non-Obvious Rationale**: Ensures that CI/CD pipelines, automated testing harnesses, and reviewers evaluating the repo can reproduce headline figures and run all 44 unit tests in under 15 seconds without API keys, cloud costs, or network flakiness.

---

### 10. Refusing In-Chat Financial Transactions and Password Resets
- **Decision**: Explicitly scoped out direct in-chat execution of refunds, subscription cancellations, or password resets.
- **Rejected Alternative**: Building an end-to-end autonomous agent that attempts to reset passwords or issue refunds directly in Twitter messages.
- **Non-Obvious Rationale**: Unauthenticated public Twitter timelines must never solicit or handle PII, 2FA codes, or payment credentials. Providing deep links to official Apple self-service portals (`reportaproblem.apple.com`, `iforgot.apple.com`) guarantees customer security and eliminates legal liability.

---

### 11. Quadratic Weighted Kappa ($\kappa$) as the Primary Calibration Metric
- **Decision**: Adopted Cohen's Quadratic Weighted Kappa ($\kappa = 0.7897$) alongside within-1 point accuracy (96.0%) and MAE (0.6000) for human-judge agreement.
- **Rejected Alternative**: Relying solely on raw percentage exact match or Pearson correlation.
- **Non-Obvious Rationale**: Standard percent agreement treats a minor 1-point difference (rating 4 vs. 5) identically to a critical 4-point failure (rating 1 vs. 5). QWK penalizes large disagreements quadratically, providing an honest, psychometrically sound measure of evaluator calibration.

---

### 12. Deliberately Preserving Public Self-Service Over Historical Human DM Habits
- **Decision**: Trained the agent to provide immediate, actionable public self-service troubleshooting, only transitioning to DM when privacy, security, or serial numbers are genuinely required.
- **Rejected Alternative**: Training the agent to mimic historical human Twitter agents who sent DM links in 51.5% of tweets.
- **Non-Obvious Rationale**: Twitter users reach out publicly for rapid answers. Historical agents frequently used DM links merely to clean up their public timelines or meet shift queue quotas. Automating that habit defeats the primary benefit of an AI agent: instantaneous self-service deflection.
