"""Step 2: inject documented, labelled anomalies into the clean data (test period only).

Usage: python src/inject_anomalies.py
Reads  data/clean_mixers.csv
Writes data/dataset_injected.csv  (obs columns + label + group) and data/anomaly_log.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED, SPLIT_FRAC = 42, 0.5            # first 50 % of time stays clean (training)
PH = ["L1", "L2", "L3"]

# (type, group, min_len, max_len, how many)   lengths in 15-min windows (96 = 1 day)
SPECS = [("shutdown", "genuine_event", 4, 16, 4), ("surge", "genuine_event", 4, 12, 2),
         ("drift", "data_quality", 192, 480, 3), ("bias", "data_quality", 96, 288, 3),
         ("ct_ratio", "data_quality", 96, 192, 2), ("stuck", "data_quality", 8, 32, 3),
         ("falsified", "data_quality", 96, 288, 3), ("phase_bias", "data_quality", 96, 288, 3),
         ("gap", "data_quality", 4, 24, 4)]


def inject(clean, seed=SEED, split_frac=SPLIT_FRAC):
    rng = np.random.default_rng(seed)
    df = clean.copy()
    df["time"] = pd.to_datetime(df["time"])
    times = np.sort(df.time.unique()); T = len(times)
    df["step"] = df.time.map({t: i for i, t in enumerate(times)})
    df = df.sort_values(["asset", "step"]).reset_index(drop=True)
    for c in ["E"] + PH:
        df[c + "_obs"] = df[c]
    df["label"], df["group"], df["event_id"] = "normal", "normal", -1
    obs = ["E_obs"] + [p + "_obs" for p in PH]
    t0 = int(T * split_frac); used = np.zeros(T, bool); assets = sorted(df.asset.unique())
    log, eid = [], 0

    def place(l):
        for _ in range(500):
            s = int(rng.integers(t0, T - l))
            if not used[max(0, s - 24): s + l + 24].any():
                used[s: s + l] = True
                return s
        return None

    for kind, group, lo, hi, n in SPECS:
        for _ in range(n):
            l = int(rng.integers(lo, hi + 1)); s = place(l)
            if s is None:
                print(f"WARNING: no room to place {kind}; use a longer period")
                continue
            a = None if group == "genuine_event" else str(rng.choice(assets))
            m = (df.step >= s) & (df.step < s + l)
            if a:
                m &= df.asset == a
            idx = df.index[m]; param = ""
            if kind == "shutdown":                                  # whole line stops
                df.loc[idx, obs] *= 0.02; param = "x0.02 on all mixers"
            elif kind == "surge":                                   # whole line surges
                df.loc[idx, obs] *= 1.6; param = "x1.6 on all mixers"
            elif kind == "drift":                                   # gradual register drift (E only)
                df.loc[idx, "E_obs"] *= 1 + np.linspace(0, 0.4, len(idx)); param = "0 -> +40 % ramp"
            elif kind == "bias":                                    # constant over-reading, all registers
                df.loc[idx, obs] *= 1.15; param = "x1.15"
            elif kind == "ct_ratio":                                # wrong CT ratio, all registers
                df.loc[idx, obs] *= 2.0; param = "x2.0"
            elif kind == "stuck":                                   # frozen sensor
                df.loc[idx, obs] = df.loc[idx[0], obs].values; param = "value frozen"
            elif kind == "falsified":                               # under-reported total only
                df.loc[idx, "E_obs"] *= 0.5; param = "total x0.5, phases untouched"
            elif kind == "phase_bias":                              # one phase over-reads
                df.loc[idx, "L1_obs"] *= 1.5
                df.loc[idx, "E_obs"] = df.loc[idx, [p + "_obs" for p in PH]].sum(axis=1); param = "L1 x1.5"
            elif kind == "gap":                                     # communication loss
                df.loc[idx, obs] = np.nan; param = "NaN"
            # label only windows where the reading really changed (an idle machine stays 'normal')
            diff = (df.loc[idx, "E_obs"] - df.loc[idx, "E"]).abs()
            eff = idx[(diff > 0.02 * df.loc[idx, "E"].abs().clip(lower=0.05)) | df.loc[idx, "E_obs"].isna()]
            df.loc[eff, ["label", "group", "event_id"]] = [kind, group, eid]
            log.append(dict(event_id=eid, type=kind, group=group, asset=a or "all",
                            start=str(times[s]), end=str(times[s + l - 1]), windows=l,
                            labelled_windows=len(eff), params=param))
            eid += 1
    return df, pd.DataFrame(log), times[t0]


if __name__ == "__main__":
    clean_path = Path("data/clean_mixers.csv")
    if not clean_path.exists():
        raise FileNotFoundError(f"{clean_path} not found. Run prepare_dataset.py first.")
    clean = pd.read_csv(clean_path)
    df, log, split = inject(clean)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/dataset_injected.csv", index=False)
    log.to_csv("data/anomaly_log.csv", index=False)
    print("split time:", split)
    print(log.to_string(index=False))
    print("\nInjected label distribution:\n", df.label.value_counts())
