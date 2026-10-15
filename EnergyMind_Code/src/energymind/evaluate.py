"""Evaluation against synthetic labels. Reports both window-level and event-level numbers, because
operators care about "did we notice the event, and how many false alarms did it cost?"."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def group_alert_runs(flag: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    """Merge alert windows into incidents: consecutive alerts separated by <= max_gap quiet windows."""
    idx = np.flatnonzero(np.asarray(flag, dtype=bool))
    if len(idx) == 0:
        return []
    runs, s, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        if i - prev - 1 > max_gap:
            runs.append((s, prev))
            s = int(i)
        prev = int(i)
    runs.append((s, prev))
    return runs


def point_metrics(y: pd.Series, flag: pd.Series, score: pd.Series | None = None) -> dict:
    y = y.astype(bool).to_numpy()
    f = flag.fillna(False).astype(bool).to_numpy()
    tp, fp = int((y & f).sum()), int((~y & f).sum())
    fn, tn = int((y & ~f).sum()), int((~y & ~f).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp and prec == prec and rec == rec else 0.0
    out = {"precision": prec, "recall": rec, "f1": f1,
           "false_alert_rate": fp / (fp + tn) if fp + tn else float("nan"),
           "alerts": int(f.sum()), "tp": tp, "fp": fp, "fn": fn}
    if score is not None and y.any() and (~y).any():
        s = score.fillna(score.min() if score.notna().any() else 0).to_numpy()
        out["pr_auc"] = float(average_precision_score(y, s))
    return out


def event_metrics(events: pd.DataFrame, labels: pd.DataFrame, flag: pd.Series,
                  tolerance: int = 4) -> tuple[pd.DataFrame, dict]:
    """An event counts as detected if any alert falls inside it (or within `tolerance` windows
    after its end). Delay = windows from event start to first alert."""
    idx = flag.index
    f = flag.fillna(False).to_numpy()
    pos = {t: i for i, t in enumerate(idx)}
    rows = []
    for _, ev in events.iterrows():
        s, e = pos[ev["start"]], pos[ev["end"]]
        hit = np.flatnonzero(f[s:min(len(f), e + 1 + tolerance)])
        rows.append({**ev.to_dict(), "detected": bool(len(hit)),
                     "delay_windows": int(hit[0]) if len(hit) else np.nan})
    per_event = pd.DataFrame(rows)

    # False incidents: runs of alerts (gap <= 2 windows) that touch no injected window (+tolerance).
    y = (labels["is_injected"] | labels.get("ambiguous", 0)).to_numpy().astype(bool)
    near = pd.Series(y).rolling(2 * tolerance + 1, center=True, min_periods=1).max().astype(bool).to_numpy()
    runs = group_alert_runs(f, max_gap=2)
    false_runs = [r for r in runs if not near[r[0]:r[1] + 1].any()]
    days = max(len(idx) / 96, 1e-9)
    summary = {
        "events": len(per_event),
        "events_detected": int(per_event["detected"].sum()) if len(per_event) else 0,
        "event_recall": float(per_event["detected"].mean()) if len(per_event) else float("nan"),
        "median_delay_windows": float(per_event["delay_windows"].median()) if len(per_event) else float("nan"),
        "false_incidents": len(false_runs),
        "false_incidents_per_day": len(false_runs) / days,
    }
    return per_event, summary
