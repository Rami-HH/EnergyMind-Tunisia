# EnergyMind Tunisia: A Trust Layer for Industrial Energy Data

**TSYP14 Technical Challenge — Phase 1 Project Proposal**  
*Track: Intelligent Energy Systems & Industrial AI Trust*  
*Team: EnergyMind Tunisia | Affiliation: IEEE PES / IES / Young Professionals (YP)*  

---

## Executive Summary

As factories and national utilities (such as STEG in Tunisia) increasingly rely on automated energy telemetry and demand-side management, an unaddressed vulnerability emerges: the assumption that meter data is consistently trustworthy. In reality, industrial meters frequently suffer from current transformer (CT) ratio misconfigurations, phase loss, sensor drift, frozen registers, communication dropouts, and even intentional tampering. If downstream automation or automated billing blindly acts on faulty data, the consequences range from unjustified penalties to catastrophic production downtime.

**EnergyMind Tunisia** introduces an intelligent AI trust layer situated between raw telemetry and industrial control systems. Rather than viewing meters in isolation, our framework harnesses **peer-asset physical correlation** across identical production units (`mix_a` to `mix_d`) to reliably distinguish **genuine operational events** (e.g., line-wide load shedding or plant surges) from **data-quality metering failures**. Furthermore, every anomaly flag is accompanied by a calibrated confidence percentile, an interpretable plain-language explanation, a strict human-in-the-loop escalation policy, and an architecture ready for privacy-preserving federated knowledge sharing ("the noosphere").

---

## 1. Scope & Industrial Production Context

### 1.1 Chosen System Scope
The system monitors an industrial manufacturing process line consisting of **four comparable heavy industrial mixers** (`mix_a`, `mix_b`, `mix_c`, `mix_d`). Telemetry is recorded at **15-minute intervals**, capturing active energy and power per electrical phase (`L1`, `L2`, `L3`) and total active energy (`ALL`).

### 1.2 Rationale: Why Multi-Mixer Peer Correlation?
In isolated single-meter anomaly detection, a sudden 90% drop in energy consumption is ambiguous: is the machine's CT sensor broken, or did the factory line stop for a scheduled shift change? 
Because the four mixers perform synchronized production tasks in the same facility:
- **Correlated Multi-Machine Shifts**: When all 4 mixers drop or spike simultaneously, it constitutes a **genuine plant or grid event**.
- **Isolated Divergence**: When a single mixer diverges sharply while its three identical peers continue running normally, it provides irrefutable physical evidence of a **local metering or sensor fault**.

### 1.3 Stand-in Context & Adaptation for Tunisia
Because public 15-minute interval industrial datasets do not currently exist for Tunisian manufacturing facilities, we utilize the high-resolution Flynn et al. manufacturing dataset (CC BY 4.0) as an empirical benchmark. The feature schema and detection pipelines are completely generalizable: they can be ingested directly into Tunisian telemetry infrastructures (such as STEG smart metering or factory SCADA systems) and tuned to local industrial tariff bands (e.g., peak/off-peak pricing) and solar generation curves (PVGIS).

---

## 2. Measurable Objectives

The Phase 1 prototype was developed and evaluated against strict, quantifiable performance criteria:
1. **Low False Positive Rate**: Achieve an empirical false positive rate (FPR) below $2.0\%$ (Phase 1 achieved **$1.35\%$**).
2. **Deterministic Event Separation**: Achieve $\ge 90\%$ accuracy in distinguishing genuine grid/plant disturbances from sensor faults (Phase 1 achieved **$100.0\%$**).
3. **Comprehensive Failure Coverage**: Successfully identify diverse industrial failure modes, achieving $100\%$ recall on drift, falsified readings, and communication gaps.
4. **Calibrated Confidence & Explainability**: Ensure $100\%$ of flagged anomalies carry a calibrated confidence percentile and an auditable plain-language explanation linking to the primary deviating feature.
5. **Real-Time Edge Throughput**: Maintain sub-millisecond per-reading inference latency (Phase 1 achieved **$0.0091\text{ ms}$ per window**), enabling direct deployment on low-cost edge gateways.

---

## 3. Data & Anomaly Simulation Protocol

### 3.1 Data Splitting
Evaluation uses a strict **chronological time-based split**:
- **Baseline Training Segment ($50\%$)**: 2025-01-01 00:00 to 2025-07-02 12:30 UTC. Retained completely pristine to establish normal seasonal profiles, inter-phase relationships, and machine peer baselines.
- **Evaluation Segment ($50\%$)**: 2025-07-02 12:45 to 2025-12-31 23:45 UTC. Targeted for documented synthetic anomaly injections.

### 3.2 Documented Anomaly Scenarios
A total of 27 distinct anomaly events across 9 failure families were injected:

| Scenario | Class | Simulation Mechanism | Industrial Precedent |
|---|---|---|---|
| **Shutdown** | Genuine Plant Event | All 4 mixers reduced to $\times 0.02$ for 1 to 4 hours | Line outage / Shift change |
| **Surge** | Genuine Plant Event | All 4 mixers scaled to $\times 1.6$ for 1 to 3 hours | Grid voltage / mechanical batch surge |
| **Drift** | Data Quality | Linear gain ramp $0 \to +40\%$ over 2–5 days | Sensor calibration decay |
| **Bias** | Data Quality | Constant over-reading ($\times 1.15$) for 1–3 days | Hardware multiplier drift |
| **CT Ratio** | Data Quality | Doubled reading ($\times 2.0$) across all phases | Misconfigured CT ratio in gateway |
| **Stuck** | Data Quality | Constant register value frozen for 2–8 hours | Firmware lockup / buffer stall |
| **Falsified** | Data Quality | Total reading halved ($\times 0.5$), phases untouched | Meter tampering / fraud |
| **Phase Bias** | Data Quality | Phase L1 scaled $\times 1.5$, total re-summed | Wiring misalignment |
| **Gap** | Data Quality | Values set to NaN for 1–6 hours | Packet loss / communication blackout |

Every event is recorded with timestamps, parameters, and affected assets in `data/anomaly_log.csv`.

---

## 4. Preliminary Technical Design: Goal 1 (Anomaly & Bias Detection)

### 4.1 Engineered Feature Space
For each mixer $i$ at window $t$, seven domain-specific features are computed:
1. `level`: Scaled active energy relative to the mixer's historical baseline mean.
2. `peer_dev`: Deviation from the concurrent median of the other mixers: $\frac{e_{i,t} - \text{median}_{j}(e_{j,t})}{\text{median}_{j}(e_{j,t}) + 0.1}$.
3. `prof_dev`: Z-score deviation against historical normal consumption for that specific hour-of-day and day-of-week: $\frac{e_{i,t} - \mu_{a,w,h}}{\sigma_{a,w,h} + 0.05}$.
4. `phase_imb`: Phase imbalance ratio: $\frac{\max(L_1, L_2, L_3) - \min(L_1, L_2, L_3)}{\sum L_k + 0.05}$.
5. `consist`: Internal energy conservation discrepancy: $\frac{|E_{\text{ALL}} - \sum_{k=1}^3 L_k|}{E_{\text{ALL}} + 0.05}$.
6. `flat`: Rolling 8-window log standard deviation: $\log_{10}(\text{std}_8 + 10^{-4})$ (detects frozen values).
7. `missing`: Binary flag indicating missing telemetry / communication failure.

### 4.2 Ensemble Architecture
- **Isolation Forest**: Fits an ensemble of 200 isolation trees on clean training features to identify multivariate anomalies and unexpected correlation breakdowns.
- **Rule-Backed Thresholds**: Non-parametric thresholds calibrated on the 99.9th percentile of clean training data for physical conservation checks (`consist`, `phase_imb`, and `prof_dev`).
- **Consensus Voting**: An alert is triggered if the Isolation Forest, domain rules, or communication gap rule detect an irregularity.
- **Genuine vs. Data Quality Separation**: If $\ge 75\%$ of monitored mixers trigger concurrent flags at time step $t$, the system classifies the incident as a **Genuine Plant/Grid Event**. If fewer than $75\%$ trigger, it is classified as an isolated **Data Quality Problem**.

---

## 5. Preliminary Technical Design: Goal 2 (Confidence & Explainability)

### 5.1 Calibrated Confidence Scoring
Rather than arbitrary heuristic weights, confidence is derived empirically from the cumulative distribution function (percentile) of isolation scores observed during clean training:
$$\text{Confidence}(s) = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \mathbb{I}(s_{\text{train}, i} \le s)$$
- Where rules are breached, confidence is anchored to $\ge 0.99$.
- Complete communication losses (`gap`) are assigned deterministic confidence of $1.00$.

### 5.2 Plain-Language Explanations & Auditability
For every flagged window, the top contributing feature is computed via Z-score distance from clean training norms. Explanations follow a human-readable template:
> `mix_a: this mixer disagrees with the other mixers (z=7.1); verdict data_quality, confidence 1.00`

Every flag is permanently preserved in `results/decision_log.csv` with fields: `time, asset, verdict, confidence, action, explanation, label`.

---

## 6. Preliminary Technical Design: Goal 3 (Human-AI Escalation)

To prevent both automation complacency (over-trust) and alarm fatigue (under-trust), our governance policy enforces clear operational boundaries:

| Condition | Automated System Action | Operator Involvement |
|---|---|---|
| **Data Quality Fault & Confidence $\ge 99.9\%$** | **`AUTO`**: Tag meter suspect, substitute synthetic peer estimate, log action | Optional retroactive audit |
| **Data Quality Fault & Confidence $< 99.9\%$** | **`HUMAN`**: Raise advisory ticket, highlight evidence | Operator reviews & signs off |
| **Genuine Event (Line Shutdown / Surge)** | **`HUMAN`**: Alert plant managers; **NEVER** automate control | Operators assess electrical grid/plant state |
| **Operator Override** | Record rationale in audit log | Feeds into weekly threshold recalibration |

---

## 7. Preliminary Technical Design: Goal 4 (The Noosphere: Privacy-Preserving Shared Learning)

Industrial facilities cannot share raw energy telemetry due to production secrecy, tariff negotiations, and competitive confidentiality. **EnergyMind's Noosphere Architecture** decouples anomaly learning from raw data:

```
[ Factory Node 1 ] ----(Model Weights & Thresholds)----> [ Central Knowledge Hub ]
[ Factory Node 2 ] ----(Anonymized Fault Signatures)---> (Federated Aggregator)
[ Factory Node 3 ] <---(Updated Global Signatures)------ [ Signed Release v1.2 ]
```

### 7.1 What Is Shared vs. What Stays Local
- **Remains Strictly On-Premise**: Raw kWh readings, load profiles, operating schedules, facility names.
- **Shared Across Federation**: Feature deviation centroids (e.g., standard drift ramp signatures), calibrated quantile cutoffs, and verified false-positive suppression rules.

### 7.2 Federated Consensus
When an anomaly signature observed at one facility (e.g., a specific CT ratio tag corruption) is encountered at a second independent node, the central aggregator validates the signature match and increases global detection confidence across all member nodes.

---

## 8. Business Scope & Cost-Benefit Analysis (Bonus Scope)

### 8.1 Implementation & Deployment Economics (Estimated for 50-Asset Facility)
- **Edge Gateway Hardware**: Industrial Raspberry Pi / Advantech edge computer (~$350 – $500 per production line).
- **Integration**: Modbus TCP/IP gateway integration and telemetry pipeline configuration (~$2,500 one-off).
- **Maintenance & Retraining**: Semi-annual model recalibration and threshold tuning (~$1,200 / year).

### 8.2 Return on Investment (ROI)
- **Prevention of STEG Reactive Power / Peak Penalties**: Detecting CT drift and phase imbalance early prevents costly billing penalties (typically $3,000 – $10,000/year for medium industrial facilities in Tunisia).
- **Avoided Downtime**: Catching motor/mixer degradation before catastrophic failure saves hours of production downtime ($5,000+ per hour in batch chemical/food manufacturing).

---

## 9. Team Structure & Roadmap to Phase 2 (December 12)

### 9.1 Team Roles & IEEE Memberships
- **Lead AI & Systems Engineer**: Model pipeline, Isolation Forest, feature engineering (IEEE Student Member).
- **Power Systems & Domain Specialist**: Industrial metering standards, STEG tariff alignment (IEEE PES / IES Member).
- **Software & UI Developer**: Streamlit operator dashboard, decision audit logging (IEEE Young Professionals).

### 9.2 Timeline to Final Submission
- **Phase 1 (October)**: Baseline 4-mixer prototype, 9-fault injection benchmark, decision log, technical proposal.
- **Milestone 2 (November 10)**: Integration of TreeSHAP explainability for continuous feature attribution.
- **Milestone 3 (November 25)**: Interactive Streamlit operator interface featuring real-time peer comparison plots and approval workflows.
- **Final Milestone (December 12)**: Multi-node federated simulation demo, comprehensive final report, and video walkthrough.
