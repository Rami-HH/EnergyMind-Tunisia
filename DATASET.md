# Dataset

## Baseline (real data)
Industrial Energy Monitoring Dataset, Flynn et al., Munster Technological University, CC BY 4.0.  
Assets used: `mix_a`, `mix_b`, `mix_c`, `mix_d`. 15-minute energy per phase (`L1`, `L2`, `L3`) and total (`ALL`).  
Modified: synthetic anomalies injected into the second half of the period (see below).

## Why this stand-in for Tunisia
No public Tunisian factory data at 15-minute resolution exists. The pipeline only needs the same columns (`asset`, `time`, energy per phase), so it can be directly retrained on STEG (Société Tunisienne de l'Electricité et du Gaz) or private factory telemetry data.  
Tunisian adaptation: rescale seasonality with STEG industrial tariff structures and monthly consumption patterns, and incorporate PVGIS solar generation profiles.

## Train/Test Split
Time-based chronological split. The first 50% of the period (2025-01-01 to 2025-07-02) is left completely clean (training baseline). All anomalies are injected exclusively into the last 50% (test evaluation period).

## Injected Anomalies
| Type | Group | What it simulates | Generation |
|---|---|---|---|
| **shutdown** | genuine event | The whole production line stops | All 4 mixers x0.02 for 4 to 16 windows |
| **surge** | genuine event | The whole production line surges | All 4 mixers x1.6 for 4 to 12 windows |
| **drift** | data quality | Register/sensor drifting | One mixer total ramps 0 to +40% over 2 to 5 days |
| **bias** | data quality | Constant over-reading | One mixer, all registers x1.15 for 1 to 3 days |
| **ct_ratio** | data quality | Wrong CT ratio configuration | One mixer, all registers x2.0 for 1 to 2 days |
| **stuck** | data quality | Frozen sensor | One mixer repeats one value for 2 to 8 hours |
| **falsified** | data quality | Under-reported total (meter tampering) | One mixer total x0.5, phases untouched, 1 to 3 days |
| **phase_bias** | data quality | One phase mis-measured | One mixer L1 x1.5, total recomputed |
| **gap** | data quality | Communication/network packet loss | One mixer NaN for 1 to 6 hours |

Random seed 42. Every injected event is recorded in `data/anomaly_log.csv`. A window is labelled anomalous only if the reading actually changed (an idle machine stays "normal").
The simulated failure modes mirror documented real-world industrial metering issues (such as CT ratio misconfigurations, phase misalignments, and data gaps documented in Flynn et al.).
