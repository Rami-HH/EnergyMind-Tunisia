# EnergyMind Tunisia - Phase 1 (code + filtered dataset)

Anomaly detection prototype for the TSYP14 EnergyMind Tunisia challenge (code only).
Chosen slice: **factory power line** (industrial 15-minute energy data).
The prototype only recommends review; it never controls equipment.

This code works on the **filtered dataset** delivered separately in `EnergyMind_Filtered_Dataset.zip`
(11 assets). These instructions are for Windows (Command Prompt).

## Step 0 - Put the dataset next to the code
Extract `EnergyMind_Filtered_Dataset.zip` and move its `dataset` folder inside `EnergyMind_Code`,
so you have:

```
EnergyMind_Code\
    dataset\          <- from the dataset zip
    src\  scripts\  notebooks\  tests\  requirements.txt  README.md
```

## Step 1 - Install Python
Install Python 3.10 or newer from python.org. During installation, tick **"Add Python to PATH"**.
Check it works:

```
python --version
```

## Step 2 - Open Command Prompt in the project folder
Open the `EnergyMind_Code` folder in File Explorer, click the address bar,
type `cmd` and press Enter.

## Step 3 - Create the environment and install the libraries
Create a Python environment:

```
python -m venv .venv
```

Activate it:

```
.venv\Scripts\activate
```

Install the required libraries:

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

These libraries provide the tools for reading the dataset, creating graphs, and detecting anomalies with machine
learning. Installation may take a few minutes.

## Step 4 - Launch the notebook
In the same Command Prompt window, enter:

```
python -m notebook
```

A browser tab opens. Open `notebooks/EnergyMind_Phase1_v2.ipynb`, then choose **Run > Run All Cells**
(or press Shift+Enter cell by cell, from top to bottom). The notebook finds the dataset in `dataset/`
automatically.

The notebook covers: data validation, the synthetic anomaly benchmark, a real-fault case study from the
dataset's quality log, and the audit log (saved as `energymind_phase1_audit_log.csv` next to the notebook).

## Optional - Check the code and run all assets
In the same Command Prompt window (environment still active):

```
pytest -q
python scripts\run_benchmark.py --out results
```

The first command runs the 13 tests. The second runs every asset in `dataset/` (about 30 seconds) and writes
CSV results to the `results` folder.

## Troubleshooting
- `'python' is not recognized`: reinstall Python and tick "Add Python to PATH".
- `.venv\Scripts\activate` is blocked in PowerShell: use Command Prompt (cmd), as above.
- `Cannot find data_csv/data_parquet`: keep the `dataset` folder next to `src`, `notebooks` and `scripts`.
- To use the full original dataset instead, set `ENERGYMIND_DATA_ROOT` to its folder before step 4
  (`set ENERGYMIND_DATA_ROOT=C:\path\to\dataset`).

## Project layout
```
dataset/                      filtered dataset (from the dataset zip)
src/energymind/data.py        load files, pivot phases, read the quality log
src/energymind/features.py    weekly baseline and phase-share features (fitted on train only)
src/energymind/synthetic.py   synthetic anomaly injection (spike_drop, gain_error, drift, phase_loss)
src/energymind/detectors.py   four named rules + Isolation Forest
src/energymind/evaluate.py    window-level and event-level metrics
src/energymind/policy.py      verdicts and per-incident audit log
src/energymind/pipeline.py    run_synthetic_benchmark(), run_real_fault_case_study()
notebooks/                    demo notebook
scripts/run_benchmark.py      multi-asset runner
tests/                        13 tests
```

## Notes
- Change `BENCH_ASSET` / `CASE_ASSET` in the notebook to try other assets (default: `u_e` and `ahu_a`).
- Results are reproducible (fixed seed 42).
- Synthetic scores only measure recovery of injected scenarios, not real-world detection.
- Dataset licence: CC BY 4.0, (c) 2026 Christopher Flynn / Munster Technological University. Keep this
  attribution if you redistribute the dataset.
