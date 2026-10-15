"""End-to-end runs used by the notebook, the CLI script and the tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import PHASES
from .detectors import RULE_NAMES, IForestDetector, RuleDetector
from .evaluate import event_metrics, point_metrics
from .features import build_features, fit_context
from .policy import build_audit_log, decide
from .synthetic import inject

DATA_COLS = [f"p_{p}" for p in (*PHASES, "ALL")] + ["rel_cov"]


def blank_out(wide: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Treat masked windows as missing (kept on the grid so diff/rolling stay correct)."""
    w = wide.copy()
    w.loc[mask, DATA_COLS] = np.nan
    w.loc[mask, "gap"] = True
    return w


def chronological_split(w: pd.DataFrame, exclude: pd.Series | None = None, frac=(0.6, 0.2, 0.2)):
    keep = w.index if exclude is None else w.index[~exclude.reindex(w.index).fillna(False).to_numpy()]
    n = len(keep)
    t1, t2 = keep[int(frac[0] * n)], keep[int((frac[0] + frac[1]) * n)]
    return w[w.index < t1], w[(w.index >= t1) & (w.index < t2)], w[w.index >= t2], (t1, t2)


@dataclass
class BenchmarkResult:
    split_times: tuple
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame            # with synthetic anomalies applied
    labels: pd.DataFrame
    events: pd.DataFrame
    feat_test: pd.DataFrame
    rules: pd.DataFrame
    if_scores: pd.Series
    if_flag: pd.Series
    decisions: pd.DataFrame
    audit_log: pd.DataFrame
    metrics: pd.DataFrame         # window-level, one row per detector
    event_table: pd.DataFrame     # event-level recall per detector and anomaly type
    event_summary: pd.DataFrame
    thresholds: dict


def run_synthetic_benchmark(wide: pd.DataFrame, asset_id: str = "asset", exclude: pd.Series | None = None,
                            seed: int = 42, n_per_type: int = 6, rule_fa: float = 0.005,
                            if_fa: float = 0.01) -> BenchmarkResult:
    w = wide if exclude is None else blank_out(wide, exclude)
    train, val, test, split_times = chronological_split(w, exclude)

    ctx = fit_context(train)                       # fitted on training only
    f_train, f_val = build_features(train, ctx), build_features(val, ctx)
    rd = RuleDetector(fa_rate=rule_fa).fit(f_train)
    fd = IForestDetector(fa_rate=if_fa, random_state=seed).fit(f_train, f_val)

    inj = inject(test, seed=seed + 1, n_per_type=n_per_type, min_load_kw=ctx.min_load_kw)
    if inj.events.empty:
        raise ValueError("No synthetic events could be injected (test segment too short, too idle, "
                         "or mostly excluded). Choose another asset or segment.")
    f_test = build_features(inj.data, ctx)
    rules = rd.apply(f_test)
    scores = fd.score(f_test)
    iflag = fd.flag(scores)

    # windows a detector can actually judge; ambiguous (partly modified, unlabelled) ones are excluded
    scored = f_test["valid"] & ~f_test["gap"] & (inj.labels["ambiguous"] == 0)
    y = inj.labels["is_injected"][scored]
    detectors = {f"rule:{r}": rules[r] for r in RULE_NAMES}
    detectors["rules (any)"] = rules.any(axis=1)
    detectors["iforest"] = iflag
    detectors["rules OR iforest"] = rules.any(axis=1) | iflag
    detectors["rules AND iforest"] = rules.any(axis=1) & iflag

    rows, ev_rows, ev_sum = [], [], []
    for name, flag in detectors.items():
        sc = scores if name == "iforest" else None
        m = point_metrics(y, flag[scored], sc[scored] if sc is not None else None)
        rows.append({"detector": name, **m})
        per_event, summ = event_metrics(inj.events, inj.labels, flag.where(scored, False))
        per_event.insert(0, "detector", name)
        ev_rows.append(per_event)
        ev_sum.append({"detector": name, **summ})
    metrics = pd.DataFrame(rows).set_index("detector")
    event_df = pd.concat(ev_rows, ignore_index=True)
    event_summary = pd.DataFrame(ev_sum).set_index("detector")
    by_type = (event_df.groupby(["detector", "anomaly_type"])["detected"].mean()
               .unstack("anomaly_type").round(2))

    decisions = decide(f_test, rules, iflag, scores)
    audit = build_audit_log(asset_id, f_test, decisions)
    return BenchmarkResult(split_times, train, val, inj.data, inj.labels, inj.events, f_test, rules,
                           scores, iflag, decisions, audit, metrics, by_type, event_summary,
                           {**rd.thresholds_, "iforest": fd.threshold_})


@dataclass
class CaseStudyResult:
    issue_start: pd.Timestamp
    feat: pd.DataFrame
    rules: pd.DataFrame
    if_flag: pd.Series
    summary: pd.DataFrame
    onset_delay_hours: dict
    decisions: pd.DataFrame
    audit_log: pd.DataFrame


def run_real_fault_case_study(wide: pd.DataFrame, log: pd.DataFrame, asset_id: str, seed: int = 42,
                              min_clean_days: int = 28, rule_fa: float = 0.005, if_fa: float = 0.01):
    """Fit on the clean period BEFORE the first logged issue, then look at what the detectors do
    from the issue start onward.

    Caveats, stated up front:
      * the log gives asset-level intervals with no timezone; we assume UTC,
      * windows inside the interval are treated as "issue present", but not every window in an
        interval need look abnormal, so recall here is a lower-bound style indication,
      * there is no clean period AFTER the issue for these assets, so false alerts cannot be
        measured out-of-sample; the alert rate on the held-out clean validation slice is shown instead,
      * choosing the pre-issue period as the baseline uses knowledge of the log; in deployment
        this corresponds to a commissioning / known-good baseline period.
    """
    rows = log[log["asset_id"] == asset_id.lower()]
    if rows.empty:
        raise ValueError(f"No logged issues for {asset_id}.")
    t0 = rows["issue_start"].min()
    pre = wide[wide.index < t0]
    if len(pre) < min_clean_days * 96:
        raise ValueError(f"{asset_id}: only {len(pre) / 96:.0f} clean days before the first issue "
                         f"(need >= {min_clean_days}).")
    cut = pre.index[int(0.7 * len(pre))]
    train, val = wide[wide.index < cut], wide[(wide.index >= cut) & (wide.index < t0)]
    ctx = fit_context(train)
    f_all = build_features(wide, ctx)
    rd = RuleDetector(fa_rate=rule_fa).fit(f_all.loc[train.index])
    fd = IForestDetector(fa_rate=if_fa, random_state=seed).fit(f_all.loc[train.index], f_all.loc[val.index])
    rules = rd.apply(f_all)
    scores = fd.score(f_all)
    iflag = fd.flag(scores)

    in_issue = pd.Series(False, index=wide.index)
    for _, r in rows.iterrows():
        in_issue |= (wide.index >= r["issue_start"]) & (wide.index <= r["issue_end"])
    scored = f_all["valid"] & ~f_all["gap"]
    det = {r: rules[r] for r in RULE_NAMES}
    det["rules (any)"] = rules.any(axis=1)
    det["iforest"] = iflag
    out, onset = [], {}
    for name, flag in det.items():
        flag = flag & scored
        vmask = scored & wide.index.isin(val.index)
        imask = scored & in_issue
        first = flag[(wide.index >= t0)]
        # onset = first window from which the detector fires on >=50% of the next 24 windows
        sustained = flag.astype(float).rolling(24, min_periods=24).mean().shift(-23) >= 0.5
        sustained = sustained[wide.index >= t0]
        onset[name] = float((sustained.idxmax() - t0).total_seconds() / 3600) if sustained.any() else np.nan
        out.append({"detector": name,
                    "alert_rate_clean_validation": float(flag[vmask].mean()),
                    "share_of_issue_windows_flagged": float(flag[imask].mean()),
                    "first_alert_after_issue_start": first.idxmax() if first.any() else pd.NaT})
    decisions = decide(f_all, rules, iflag, scores)
    audit = build_audit_log(asset_id, f_all, decisions.loc[wide.index >= t0])
    return CaseStudyResult(t0, f_all, rules, iflag, pd.DataFrame(out).set_index("detector"), onset,
                           decisions, audit)
