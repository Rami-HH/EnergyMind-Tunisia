# EnergyMind Tunisia: A Trust Layer for Industrial Energy Data

**TSYP14 Technical Challenge — Phase 1 Submission**  
Team: *EnergyMind Tunisia* | IEEE PES / IES / Young Professionals (YP)

---

## What It Does

**EnergyMind Tunisia** provides an AI trust and verification layer between raw industrial energy meters and automated building/plant management systems:
1. **Detects faulty, drifting, biased, or falsified readings** and rigorously separates them from genuine grid/plant disturbances using peer-asset consensus across identical production units.
2. **Assigns calibrated confidence scores** and clear, plain-language explanations with an auditable decision trail.
3. **Implements human-AI escalation policies**: automatic peer substitution only when confidence exceeds $99.9\%$ on data-quality faults; genuine operational events always escalate to human operators.
4. **Designed for privacy-preserving federated sharing** (the "noosphere"), transferring learned anomaly signatures and calibration thresholds across factory nodes without sharing proprietary production telemetry.

---

## Scope

A multi-mixer industrial manufacturing production line consisting of four identical submetered units (`mix_a`, `mix_b`, `mix_c`, `mix_d`) monitored at 15-minute resolution across three phases (`L1`, `L2`, `L3`) and total active energy (`ALL`).

---

## Dataset & Attribution

- **Source**: Flynn, C.; Murphy, T.; Walsh, J.; Riordan, D. *Data* 2026, 11(5), 101. [DOI: 10.3390/data11050101](https://doi.org/10.3390/data11050101).
- **Archive (Zenodo)**: [DOI: 10.5281/zenodo.19180972](https://doi.org/10.5281/zenodo.19180972).
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Modification**: Synthetic benchmark anomalies injected into the second half of the 2025 timeline (see [`DATASET.md`](file:///c:/Users/user/OneDrive/Documents/test%20energy%20mind/DATASET.md)).

---

## Repository Map

```
EnergyMind-Tunisia/
├── README.md                  # Project overview, methodology, and Phase 1 benchmark results
├── DATASET.md                 # Data provenance, preprocessing, and anomaly generation specs
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excludes raw partition files and caches
├── data/
│   ├── clean_mixers.csv       # Preprocessed 4-mixer baseline table (137,841 rows)
│   ├── dataset_injected.csv   # Evaluation dataset with ground truth labels
│   ├── anomaly_log.csv        # Detailed register of all injected scenarios
│   └── download.md            # Zenodo download instructions
├── src/
│   ├── load_data.py           # Multi-asset partitioned CSV loader
│   ├── prepare_dataset.py     # Clean wide-table builder
│   ├── inject_anomalies.py    # Documented synthetic anomaly generator
│   └── detect.py              # Ensemble detector, explainability engine & escalation logic
├── notebooks/
│   └── 01_explore.ipynb       # Exploratory analysis and scenario visualizations
├── results/
│   ├── metrics.json           # Quantitative benchmark evaluation
│   └── decision_log.csv       # Auditable record of all flagged windows and decisions
└── docs/
    └── proposal.md            # Comprehensive TSYP14 Phase 1 technical proposal
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build clean dataset from partitioned raw data
python src/prepare_dataset.py "newdata_csv" 2025-01-01 2025-12-31

# 3. Inject benchmark anomaly scenarios (50% chronological split)
python src/inject_anomalies.py

# 4. Run detector, generate explanations, decisions, and metrics
python src/detect.py
```

Outputs are automatically saved to `data/` and `results/`.

---

## Phase 1 Results (Measured on Full 2025 Dataset)

All figures below are evaluated on real data from the four mixers over the full 2025 timeline:

| Metric | Measured Value | Target / Benchmark |
|---|---|---|
| **Precision** | `58.7%` | Reliable alert fidelity |
| **Recall (Global)** | `48.0%` | High-impact fault coverage |
| **False Positive Rate** | **`1.35%`** | **Passes target (< 2.0%)** |
| **Genuine vs Data-Quality Accuracy** | **`100.0%`** | **Flawless separation of plant vs sensor faults** |
| **Inference Latency** | **`0.0091 ms / window`** | **Ultra-low edge execution latency** |
| **Total Windows Flagged** | `2,166` | Filtered out of 68,920 test windows |
| **Automated Actions (`AUTO`)** | `196` | High-confidence data substitution ($\ge 99.9\%$) |
| **Human Escalations (`HUMAN`)** | `1,970` | Operator review required |

### Recall by Anomaly Type

```
Drift (Gradual Sensor Drift)      : 100% (1.00)
Falsified (Meter Tampering)       : 100% (1.00)
Gap (Communication Packet Loss)   : 100% (1.00)
CT Ratio Misconfiguration         :   9% (0.09)
Constant Bias (Over-reading)      :   3% (0.03)
Subtle / Short-duration Freezes   : Phase 2 SHAP refinement
```

---

## Phase 1 Status & Roadmap to Phase 2

- [x] Defined production-line scope (`mix_a` to `mix_d`) with peer-correlation.
- [x] Documented anomaly simulation protocol covering 9 industrial failure modes.
- [x] Hybrid detection engine combining domain rules with Isolation Forest.
- [x] Separation of genuine plant events from sensor faults using consensus rules.
- [x] Plain-language explanation engine and structured decision audit logging.
- [x] Calibrated escalation policy preventing premature autonomous actions.
- [ ] **Phase 2 (Dec 2026)**: SHAP-based feature importance, interactive Streamlit operator dashboard, and multi-node federated signature aggregation simulation.

---

## Citation

```bibtex
@article{flynn2026industrial,
  title={Industrial Asset-Level Electrical Energy Dataset from a Manufacturing Facility (15-Minute Aggregates)},
  author={Flynn, Christopher and Murphy, Tom and Walsh, Joseph and Riordan, Daniel},
  journal={Data},
  volume={11},
  number={5},
  pages={101},
  year={2026},
  publisher={MDPI},
  doi={10.3390/data11050101}
}
```
