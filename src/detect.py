"""Step 3: detect, classify (genuine event vs data-quality), explain, escalate, evaluate.

Usage: python src/detect.py
Reads  data/dataset_injected.csv (or injects on the fly from clean_mixers.csv)
Writes results/decision_log.csv, results/metrics.json
"""
import json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATS = ["level", "peer_dev", "prof_dev", "phase_imb", "consist", "flat", "missing"]
TEXT = {"level": "the reading level is unusual for this meter",
        "peer_dev": "this mixer disagrees with the other mixers",
        "prof_dev": "consumption is far from this mixer's normal pattern for this hour and weekday",
        "phase_imb": "the three phases are unbalanced",
        "consist": "the total does not equal L1+L2+L3",
        "flat": "the reading is frozen / unnaturally flat",
        "missing": "the reading is missing (communication loss)"}
GENUINE_SHARE = 0.75          # >= 75 % of mixers flagged together -> genuine grid/plant event
AUTO_CONF = 0.999             # auto-action only above this confidence
RULE_FEATS = ["prof_dev", "consist", "phase_imb"]   # rule-backed checks (data-driven thresholds)


def build_features(df, split_time):
    df = df.sort_values(["step", "asset"]).reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    split_time = pd.to_datetime(split_time)
    tr = df.time < split_time
    cols = ["E_obs", "L1_obs", "L2_obs", "L3_obs"]
    df["missing"] = df.E_obs.isna().astype(int)
    f = df[cols].fillna(0)
    scale = f.E_obs[tr].groupby(df.asset[tr]).mean().clip(lower=1e-6)
    en = f.E_obs / df.asset.map(scale)
    med = en.groupby(df.step).transform("median")
    df["level"] = en.clip(upper=10)
    df["peer_dev"] = (en - med) / (med + 0.1)
    df["hour"], df["wd"] = df.time.dt.hour, df.time.dt.dayofweek
    prof = pd.DataFrame({"en": en[tr], "a": df.asset[tr], "w": df.wd[tr], "h": df.hour[tr]}) \
        .groupby(["a", "w", "h"]).en.agg(["mean", "std"])
    pm = prof.reindex(pd.MultiIndex.from_arrays([df.asset, df.wd, df.hour])).reset_index(drop=True)
    df["prof_dev"] = ((en - pm["mean"].fillna(en[tr].mean())) / (pm["std"].fillna(1) + 0.05)).clip(-20, 20)
    ph = f[["L1_obs", "L2_obs", "L3_obs"]]
    df["phase_imb"] = (ph.max(axis=1) - ph.min(axis=1)) / (ph.sum(axis=1) + 0.05)
    df["consist"] = (f.E_obs - ph.sum(axis=1)).abs() / (f.E_obs.abs() + 0.05)
    rs = en.groupby(df.asset).transform(lambda s: s.rolling(8, min_periods=1).std()).fillna(0.1)
    df["flat"] = np.log10(rs + 1e-4)                      # very negative = frozen
    return df, tr


def run(df, split_time):
    df, tr = build_features(df, split_time)
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=0).fit(df.loc[tr, FEATS])
    train_scores = np.sort(-iso.score_samples(df.loc[tr, FEATS]))
    mu, sd = df.loc[tr, FEATS].mean(), df.loc[tr, FEATS].std() + 1e-9
    thr = df.loc[tr, RULE_FEATS].abs().quantile(0.999) * 1.5     # rule thresholds from clean data only
    te = df[~tr].copy()

    t0 = time.perf_counter()
    te["score"] = -iso.score_samples(te[FEATS])
    latency_ms = (time.perf_counter() - t0) / len(te) * 1000
    rule = (te[RULE_FEATS].abs() > thr).any(axis=1)                                 # interpretable rules
    te["flag"] = (iso.predict(te[FEATS]) == -1) | rule | (te.missing == 1)         # ensemble: IF + rules + gap rule
    te["confidence"] = np.searchsorted(train_scores, te.score) / len(train_scores)  # percentile vs normal
    te.loc[te.missing == 1, "confidence"] = 1.0
    te.loc[rule, "confidence"] = np.maximum(te.loc[rule, "confidence"], 0.99)        # rule hit = strong evidence

    share = te.groupby("step").flag.transform("mean")
    te["verdict"] = np.where(~te.flag, "normal", np.where(share >= GENUINE_SHARE, "genuine_event", "data_quality"))

    z = ((te[FEATS] - mu) / sd).abs()
    te["top_feature"] = z.idxmax(axis=1)
    te.loc[te.missing == 1, "top_feature"] = "missing"
    log = te[te.flag].copy()
    log["explanation"] = [f"{a}: {TEXT[f]} (z={z.loc[i, f]:.1f}); verdict {v}, confidence {c:.2f}"
                          for i, a, f, v, c in zip(log.index, log.asset, log.top_feature, log.verdict, log.confidence)]
    log["action"] = np.where((log.verdict == "data_quality") & (log.confidence >= AUTO_CONF),
                             "AUTO: mark meter suspect, substitute peer estimate",
                             np.where(log.verdict == "genuine_event", "HUMAN: confirm real event (never auto)",
                                      "HUMAN: low confidence, operator review"))
    return te, log, latency_ms


def evaluate(te, latency_ms):
    y = te.label != "normal"
    tp, fp, fn, tn = (te.flag & y).sum(), (te.flag & ~y).sum(), (~te.flag & y).sum(), (~te.flag & ~y).sum()
    both = te.flag & y
    return {"precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
            "false_positive_rate": round(fp / max(fp + tn, 1), 4),
            "genuine_vs_dq_accuracy": round(float((te.verdict[both] == te.group[both]).mean()), 3),
            "latency_ms_per_row": round(latency_ms, 4),
            "recall_by_type": te[y].groupby("label").flag.mean().round(2).to_dict(),
            "flagged_windows": int(te.flag.sum())}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    clean_path = Path("data/clean_mixers.csv")
    injected_path = Path("data/dataset_injected.csv")
    
    if injected_path.exists():
        df = pd.read_csv(injected_path)
        # derive split_time as 50% chronological point
        times = np.sort(pd.to_datetime(df["time"]).unique())
        split = times[len(times) // 2]
    elif clean_path.exists():
        from inject_anomalies import inject
        clean = pd.read_csv(clean_path)
        df, _, split = inject(clean)
    else:
        raise FileNotFoundError("Neither data/dataset_injected.csv nor data/clean_mixers.csv exists. Run prepare_dataset.py first.")

    te, log, lat = run(df, split)
    m = evaluate(te, lat)
    m["auto_actions"] = int(log.action.str.startswith("AUTO").sum())
    m["human_reviews"] = int(log.action.str.startswith("HUMAN").sum())
    Path("results").mkdir(exist_ok=True)
    log[["time", "asset", "verdict", "confidence", "action", "explanation", "label"]].to_csv("results/decision_log.csv", index=False)
    json.dump(m, open("results/metrics.json", "w"), indent=2)
    print("=== Phase 1 Evaluation Metrics ===")
    print(json.dumps(m, indent=2))
