# Intent Taxonomy & Codebook: AppleSupport

## 1. Provenance: Derived From Real Data, Not Banking77

While benchmark datasets like **Banking77** provide granular taxonomies for retail banking operations (wire transfers, PIN resets, chargebacks, card declines), they are **structurally unsuited** for consumer electronics and digital platform support. 

Over 90% of `@AppleSupport` interactions center on:
- Hardware malfunctions, peripheral accessories, and battery degradation.
- Operating system release cycles, update regressions, and app crashes.
- Ecosystem cloud services (iCloud), digital media subscriptions (Apple Music, App Store), and Apple ID credentials.

This taxonomy is derived empirically through stratified sampling across 5,000 real `@AppleSupport` conversational threads reconstructed from Kaggle `twcs.csv`, spanning multi-turn dialogues, simple inquiries, and escalated edge cases across the start, middle, and end of the dataset.

---

## 2. Intent Taxonomy Overview

The taxonomy consists of **8 domain-specific technical/service intents**, **1 ambiguous intake intent**, and **1 catch-all fallback intent** (10 total classes):

| Intent Code | Category | Typical Routing | Primary Resolution Pattern |
|:---|:---|:---:|:---|
| `battery_power_issue` | Battery & Power | `auto_handle` | Battery health checks, background refresh tips, service link |
| `performance_crash_freeze` | Performance & Stability | `auto_handle` | Force restart steps, storage audit, app updates |
| `software_update_glitch` | OS & Software Updates | `auto_handle` | Point-update recommendation, known bug workarounds |
| `connectivity_network_issue` | Network & Wireless | `auto_handle` | Reset network settings, carrier APN, toggle airplane mode |
| `hardware_physical_accessory` | Hardware, Screen & Audio | `escalate_human` | Genius Bar booking, mail-in repair quote, hardware diagnostics |
| `account_access_security` | Apple ID & Security | `escalate_human` | Two-factor recovery link, iForgot portal, private DM transfer |
| `billing_purchases_subscriptions` | Billing & Subscriptions | `escalate_human` | Reportaproblem.apple.com link, payment method audit, invoice dispute |
| `apps_feature_howto` | First-Party Apps & Features | `auto_handle` | User guide steps, settings toggle walkthrough, feature FAQ |
| `vague_complaint_unclear` | Ambiguous / Intake | `auto_handle` (Clarify) | Device model & OS version probing question |
| `other` | Out-of-Scope / Non-Support | `auto_handle` / Route | Language redirect, store locator link, conversation closing |

---

## 3. Ground-Truth Labeling Rules

When annotating customer interactions, human labelers and automated classifiers must adhere strictly to these 5 rules:

1. **Inbound Primacy**:
   - Intent is assigned **strictly from the customer's first inbound message**, never from Apple's historical reply or subsequent follow-up turns. Label what the customer expressed, not how the agent answered.
2. **Channel Transition vs. Intent Separation**:
   - Mentions such as *"Already in DM"*, *"Check your DM"*, or Apple's canned *"We've got your DM and will continue there"* govern the **escalation decision** (`escalate = True`, `escalate_reason = "channel_transition"`). They do **not** dictate `intent = other`.
   - The intent must still reflect the customer's stated issue (e.g. `connectivity_network_issue` if they mentioned Wi-Fi in the tweet, or `vague_complaint_unclear` if they gave zero issue details). Only label `intent = other` if the customer message is truly out of technical scope (e.g. pure greeting, thanks, or store locator).
3. **Mixed Message Precedence Hierarchy**:
   - When a customer message spans multiple issues or symptoms, resolve conflicts using this strict priority order:
     $$\text{account\_access\_security} > \text{billing\_purchases\_subscriptions} > \text{hardware\_physical\_accessory} > \text{Specific Symptoms} > \text{vague\_complaint\_unclear}$$
   - *Example*: *"My phone died and now my Apple ID is locked"* $\rightarrow$ `account_access_security` (security takes precedence over battery).
   - *Example*: *"I dropped my phone, the screen cracked, and battery drains fast"* $\rightarrow$ `hardware_physical_accessory` (physical damage takes precedence over battery).
4. **Symptom & Update Rule**:
   - If the customer names **any specific symptom** (e.g. battery drain, Wi-Fi drops, frozen screen, app crash) or **any operating system update / version** (e.g. iOS 11, new update), **do NOT** classify as `vague_complaint_unclear`. Map it to the relevant symptom or update intent.
5. **Vague Intake & Routing Alignment**:
   - `vague_complaint_unclear` is routed to **`auto_handle` by default**, where the agent asks an automated clarifying diagnostic question (*"Which device model and iOS version are you currently running?"*).
   - Escalate `vague_complaint_unclear` **only** if:
     - The tweet contains only an image/screenshot link with no accompanying explanatory text (`missing_context_screenshot`).
     - There is an explicit legal threat or physical safety hazard (`safety_legal_hazard`).
     - A channel transition / DM interaction is already active (`channel_transition`).

---

## 4. Intent Definitions, Rules & Real Grounded Examples

### 1. `battery_power_issue`
- **Definition**: Customer reports abnormal battery drain, unexpected shutoffs at non-zero percentage, device heating up during charging, or charging failures.
- **Include**: Battery percentage dropping rapidly, phone dying at 20–30%, charging cable not recognized, device overheating while charging.
- **Exclude**: Device completely dead/black screen following a physical drop (use `hardware_physical_accessory`); slow UI without battery mention (use `performance_crash_freeze`).
- **Real Examples from Data**:
  - `twcs_apple_00842`:  
    *Customer*: `"@AppleSupport Having some battery issues. Any way to check it?"`  
    *Brand*: `"Hey there! ... Please DM us which Apple device you're using. We look forward to speaking with you soon."`
  - `twcs_apple_04630`:  
    *Customer*: `"@AppleSupport upgraded to 11.02 and now that battery literally counts down before my eyes, and then stays on 1% for 20 mins b4 power off!"`  
    *Brand*: `"We've got your back. Please DM us the device you're using. We'll figure this out together."`

---

### 2. `performance_crash_freeze`
- **Definition**: Customer reports system lag, UI unresponsiveness, freezing on home screen/menus, or third-party/system apps repeatedly crashing.
- **Include**: Touch screen stutter, apps terminating upon launch, spinning wheel delays, phone frozen on an app.
- **Exclude**: Autocorrect character replacement bug (use `software_update_glitch`); device freezing on boot screen after hardware drop (use `hardware_physical_accessory`).
- **Real Examples from Data**:
  - `twcs_apple_00146`:  
    *Customer*: `"My phone been fucking up all week @115858! ... The whole phone will just crash or freeze for over a minute. It be spazzing"`  
    *Brand*: `"Thanks. We'll help get to the bottom of this together. What version of iOS are you using?"`
  - `twcs_apple_03142`:  
    *Customer*: `"Yo @115858 fix your shit, ever since IOS11 my phone has been crashing on every app."`  
    *Brand*: `"Let's make sure all your apps are updated. Tap the App Store > Updates. Install any there. Let us know the result."`

---

### 3. `software_update_glitch`
- **Definition**: Customer reports a software defect, UI anomaly, or known bug introduced by a recent operating system update or beta installation.
- **Include**: The letter "I" autocorrect / question-mark box bug, blank lock screen/notification curtain after update, calculator arithmetic delays, downgrading from public beta.
- **Exclude**: General slow battery drain after update without UI bugs (use `battery_power_issue`); physical hardware component failure (use `hardware_physical_accessory`).
- **Real Examples from Data**:
  - `twcs_apple_00003`:  
    *Customer*: `"Hey @AppleSupport and anyone else who upgraded to ios11.1, are y’all having issues with capital “I️” in the Mail app? As it puts in “A”?"`  
    *Brand*: `"@115856 Hey, let's work together to figure out what's going on. Meet us in DM and we'll continue from there."`
  - `twcs_apple_03926`:  
    *Customer*: `"@115858 after iOS 11 and the patch update, this is what my lock screen and notification menu (when I slide down) looks like..explain please.."`  
    *Brand*: `"Let's get your Lock screen back. Have you restarted since updating? Which iOS patch update?"`

---

### 4. `connectivity_network_issue`
- **Definition**: Customer reports issues establishing or maintaining network connections, including Wi-Fi drops, Bluetooth pairing, cellular data loss, or SIM card errors.
- **Include**: Wi-Fi disconnects repeatedly, unable to join public captive portal, Bluetooth won't discover accessories, "No SIM card installed" alert.
- **Exclude**: AirPod physical earbud hardware dead (use `hardware_physical_accessory`); Apple ID login network timeout (use `account_access_security`).
- **Real Examples from Data**:
  - `twcs_apple_01501`:  
    *Customer*: `"@AppleSupport Good morning! My iPhone has been unable to join a “captive Wi-Fi network” like you find in many public coffee shops. Can you help?"`  
    *Brand*: `"We're happy to help get this Wi-Fi issue taken care of. Can you tell us which iOS version you are currently using please?"`
  - `twcs_apple_01874`:  
    *Customer*: `"@AppleSupport 5 Times a day my phone says “no sim” & I have to restart it to make it go away, do I need a new phone"`  
    *Brand*: `"Hello. We are here to help. How long has this been occurring? Have their been any recent changes to your device?"`

---

### 5. `hardware_physical_accessory`
- **Definition**: Customer inquires about physical damage, display abnormalities, cracked glass, speaker/microphone failure, peripheral hardware (AirPods/chargers), or repair pricing.
- **Include**: Vertical lines or discoloration on display, broken back glass, liquid spills, single AirPod won't charge/work, Apple Watch crown failure, Genius Bar appointments.
- **Exclude**: Touchscreen unresponsive due to OS freeze (use `performance_crash_freeze`); software battery drain (use `battery_power_issue`).
- **Real Examples from Data**:
  - `twcs_apple_00265`:  
    *Customer*: `"@AppleSupport the screen on our iPad Pro has stopped working 😔. I've read a few articles about weak screens on some models - is there anywhere I can check the model number, or will it be a £400 repair bill? Thanks"`  
    *Brand*: `"Thank you for reaching out to us. We'd be happy to try everything possible to provide the best support options... DM us if you noticed any changes before the screen went black."`
  - `twcs_apple_04121`:  
    *Customer*: `"@AppleSupport My left AirPod seems to have died in the wake of 11.0.2. Restarting AirPods, re-pairing AirPods, and resetting network settings have done nothing. I bought these last week. Help!"`  
    *Brand*: `"We're here to help. Please DM us via the following link and we'll help you further from there."`

---

### 6. `account_access_security`
- **Definition**: Customer reports problems logging into Apple ID, recovering forgotten credentials/passwords, bypassing two-factor authentication, suspicious login alerts, or restoring backups.
- **Include**: Locked Apple ID, forgot security questions, trusted phone number changed, verification code not received, lost data while restoring iCloud backup.
- **Exclude**: Accidental purchase made on account (use `billing_purchases_subscriptions`); changing App Store region balance (use `billing_purchases_subscriptions`).
- **Real Examples from Data**:
  - `twcs_apple_01899`:  
    *Customer*: `"@AppleSupport I don't have access to the security questions or emails so I'm too scared to update my phone but it keeps dying on 100%? Help?"`  
    *Brand*: `"Hey there! We’d be happy to help. Send us a DM so we can look into it together"`
  - `twcs_apple_04227`:  
    *Customer*: `"@AppleSupport will I ever receive a email from Apple about a possible compromised login? probably a phishing scam just checking to be sure"`  
    *Brand*: `"We would like to take a look at this further for you. Reach out to us in DM with more information."`

---

### 7. `billing_purchases_subscriptions`
- **Definition**: Customer inquires about charges, disputes duplicate transactions, requests refunds for in-app or digital media purchases, cancels recurring subscriptions, or asks about AppleCare transfer.
- **Include**: Accidental iTunes purchase, unauthorized subscription renewal, refund status, declining credit card, transferring AppleCare+ warranty.
- **Exclude**: Hardware out-of-warranty repair quote (use `hardware_physical_accessory`); trade-in sales inquiry (use `other`).
- **Real Examples from Data**:
  - `twcs_apple_00554`:  
    *Customer*: `"@AppleSupport I just sent a dm about an accident on iTunes I need help removing the payment please."`  
    *Brand*: `"This article can help: https://t.co/Y0YoIRcZa7 If you run into issues, reach out to iTunes billing here: https://t.co/SDIe7UiyJN"`
  - `twcs_apple_02045`:  
    *Customer*: `"@AppleSupport if I sell an apple product can the AppleCare+ be passed over to the buyer?"`  
    *Brand*: `"This article may help: https://t.co/8CjEPFkTCs You can reach out with questions here: https://t.co/IBIY3vMgPj"`

---

### 8. `apps_feature_howto`
- **Definition**: Customer asks how to configure or use a built-in iOS/macOS feature, or reports functional usage issues in native apps (Apple Music, Mail, iMessage, Watch Workout, Podcasts).
- **Include**: How to delete vs archive in Apple Mail, songs not saving in Apple Music library, Apple Watch workout not counting exercise minutes, iMessage toggle stuck on SMS.
- **Exclude**: Operating system system-wide crash or freeze (use `performance_crash_freeze`); third-party app crash e.g. Spotify (use `performance_crash_freeze`).
- **Real Examples from Data**:
  - `twcs_apple_03250`:  
    *Customer*: `"@AppleSupport how can I send emails to the trash now instead of the archived option in the Apple Mail App"`  
    *Brand*: `"That's a great question! Let's look into this together. Could you tell us which Apple device you're currently using?"`
  - `twcs_apple_03503`:  
    *Customer*: `"My @115858 watch Stand function has stopped working..."`  
    *Brand*: `"When did you first encounter the issue? ... From the Watch app on your iPhone go to My Watch > Passcode > Wrist Detection."`

---

### 9. `vague_complaint_unclear`
- **Definition**: Customer expresses generic dissatisfaction or anger without identifying a specific symptom, device model, or technical context.
- **Include**: "My phone is broken", "Fix this piece of shit", "Worst customer service ever", "I hate Apple", pure one-liner complaints without symptoms.
- **Exclude**: Any message naming a symptom or update (e.g. "battery dying" $\rightarrow$ `battery_power_issue`; "update is slow" $\rightarrow$ `performance_crash_freeze`).
- **Typical Routing**: `auto_handle` — AI auto-replies with a standard clarifying probing question: *"We'd like to help. Can you let us know what model you have and what iOS version is installed?"*
- **Real Examples from Data**:
  - `twcs_apple_00006`:  
    *Customer*: `"@115861 @115862 @AppleSupport I️ upgraded. I️t didn’t work."`  
    *Brand*: `"Reach out to us via DM, and we can take a look at this with you."`
  - `twcs_apple_02153`:  
    *Customer*: `"I’m tired of this glitchy ass phone!!! @115858 @ATT"`  
    *Brand*: `"We understand that your phone is expected to work without flaws. Can you let us know the device model you are using?"`

---

### 10. `other`
- **Definition**: Interactions genuinely outside technical support and troubleshooting (non-English text, store directory requests, presales buying advice, or conversational ticket closures).
- **Include**: Non-English messages, "Where is the Apple Store in Buenos Aires?", "Should I buy an iPad or iPhone?", thank-you messages closing an interaction.
- **Exclude**: Technical support issues written in English (use 1–8); channel transitions for technical issues (use rules in Section 3).
- **Real Examples from Data**:
  - `twcs_apple_01395` (Non-English):  
    *Customer*: `"+++ Siz de benzer şekilde sorun yaşıyorsanız lütfen 'High Sierra random display sleep' başlığına şikayet yazın. https://t.co/LLu39AV5K2 @AppleSupport"`  
    *Brand*: `"We offer support via Twitter in English. Contact us for help in your preferred language here: https://t.co/IBIY3vMgPj"`
  - `twcs_apple_04161` (Store Locator / Directory):  
    *Customer*: `"@AppleSupport I have a problem with my iphone in argentina. Where can I find the address of the official apple support in Buenos Aires?thnks"`  
    *Brand*: `"This link will help you find service locally: https://t.co/IBIY3vMgPj What's going on? Maybe we can help here!"`
  - `twcs_apple_01230` (Closing / Courtesy Acknowledgment):  
    *Customer*: `"@AppleSupport Nope looks normal again!"`  
    *Brand*: `"Great! We're happy to see that. Feel free to reach out again if you need us. We're here for you. Enjoy your day!"`

---

## 5. Escalation Policy & Triage Framework

In enterprise support automation, **knowing when to yield to human agents** is vital to brand trust and security. The AI agent outputs a binary routing decision (`auto_handle` vs `escalate_human`) and a structured `escalate_reason`.

### 5.1 Auto-Handle Criteria
Tickets are safe for automated resolution when:
1. **Deterministic Troubleshooting**: Issue has standard troubleshooting trees (restarting, resetting network settings, checking battery health, installing point updates).
2. **Official Knowledge Base**: Official Apple Support articles (`support.apple.com`) directly address the issue.
3. **Feature How-To**: The query asks for configuration guidance ("How do I...") with standardized clicks.
4. **Intake Clarification**: The query is `vague_complaint_unclear` and requires an automated intake question to determine the device and OS version.

### 5.2 Mandatory Human Escalation Criteria & Stated Reasons

| # | Escalation Trigger | Applicable Intents | Stated Reason Code (`escalate_reason`) | Rationale & Policy Requirement |
|:---|:---|:---|:---|:---|
| **1** | **Account Security & Credential Recovery** | `account_access_security` | `account_security_credentials` | Password resets, 2FA bypass, locked Apple ID, and compromised accounts require out-of-band identity verification; unsafe for automated public handling. |
| **2** | **Financial Transactions & Billing Claims** | `billing_purchases_subscriptions` | `financial_billing_transaction` | Processing refund claims, billing disputes, and card declinations requires access to PCI-DSS payment gateways and private account data. |
| **3** | **Hardware Damage & Safety Hazards** | `hardware_physical_accessory`, select `battery_power_issue` | `physical_hardware_safety` | Swollen batteries, shattered screens, motherboard liquid damage, or charging smoke require physical Genius Bar inspection or mail-in repair dispatch. |
| **4** | **Legal Threats & Consumer Rights** | Any | `legal_regulatory_dispute` | Formal consumer rights invocations (e.g. EU Sale of Goods Act, legal counsel threats) must bypass AI directly to specialized escalation reps. |
| **5** | **Active Channel Transition (Already in DM)** | Any (e.g. `vague_complaint_unclear`) | `channel_transition` | When the customer or brand indicates private DM conversation is underway (e.g. `twcs_apple_00342`), automated public replies cause agent collision and confusion. |
| **6** | **Multimodal / Missing Context** | `vague_complaint_unclear` | `missing_context_screenshot` | The customer provided only a media link/screenshot with no accompanying text, preventing reliable NLP intent extraction without human review. |

---

## 6. What We Chose NOT to Model (Scope Boundaries)

1. **No Granular Per-Device Intent Splitting**:
   - We consciously rejected splitting intents into `iphone_battery_issue`, `macbook_battery_issue`, and `watch_battery_issue`. While hardware differs, the conversational intent, triage flow, and diagnostic logic share the same resolution archetype. Device type is treated as contextual entity extraction rather than intent taxonomy explosion.
2. **No Ephemeral Version-Specific Intents**:
   - We did not create dedicated intents for specific update releases (e.g. `ios_11_1_update_issue` or `ios_11_0_2_glitch`). OS versions change weekly; baking version strings into intent labels produces rapid model obsolescence. Instead, `software_update_glitch` abstracts over update regressions generically.
3. **No Sentiment-Based Intents**:
   - We do not treat `angry_customer` as an intent. Customer sentiment/frustration is an orthogonal property that modulates escalation priority, not the functional topic of the inquiry.
4. **No Direct Twitter DM Actioning**:
   - The agent drafts public Twitter replies. It does not attempt to execute private direct messages or perform transactional API operations directly into Apple internal CRM systems without human authorization.
