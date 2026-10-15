import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energymind.data import PHASES, to_wide  # noqa: E402
from energymind.detectors import RuleDetector  # noqa: E402
from energymind.evaluate import event_metrics, group_alert_runs, point_metrics  # noqa: E402
from energymind.features import build_features, fit_context  # noqa: E402
from energymind.pipeline import chronological_split, run_synthetic_benchmark  # noqa: E402
from energymind.policy import build_audit_log, decide  # noqa: E402
from energymind.synthetic import inject  # noqa: E402


def make_wide(days=120, seed=0, shares=(0.3, 0.3, 0.4)):
    """Weekly-periodic load with noise, three phases, regular grid."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=days * 96, freq="15min", tz="UTC")
    hour = idx.hour + idx.minute / 60
    weekday = (idx.dayofweek < 5).astype(float)
    total = 2 + 8 * weekday * ((hour > 7) & (hour < 18)) + rng.normal(0, 0.3, len(idx))
    total = np.clip(total, 0, None)
    w = pd.DataFrame(index=idx)
    for ph, s in zip(PHASES, shares):
        w[f"p_{ph}"] = total * s * (1 + rng.normal(0, 0.02, len(idx)))
    w["p_ALL"] = w[[f"p_{p}" for p in PHASES]].sum(axis=1)
    w["rel_cov"] = 100.0
    w["gap"] = False
    w.index.name = "window_start_utc"
    return w


def make_long(w):
    rows = []
    for ph in (*PHASES, "ALL"):
        d = pd.DataFrame({"AssetId": "x", "Phase": ph,
                          "window_start_utc": w.index.strftime("%Y-%m-%d %H:%M:%S"),
                          "Demand_kW": w[f"p_{ph}"].to_numpy(),
                          "ReliableCoveragePct": 100.0, "IsReliableWindow": 1})
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


# ---------- data ----------
def test_to_wide_pivots_phases_and_fills_grid():
    w = make_wide(days=3)
    long = make_long(w).sample(frac=1.0, random_state=1)           # shuffled
    long = long[~((long["Phase"] == "L3") & (long["window_start_utc"] > "2025-01-02"))]  # L3 vanishes
    long = long[long["window_start_utc"] != "2025-01-01 12:00:00"].copy()                # a gap
    out, rep = to_wide(long)
    assert out.index.is_monotonic_increasing and out.index.freq is not None
    assert rep["gap_windows"] == 1 and out["gap"].sum() == 1
    assert out.loc["2025-01-02 12:00", "p_L3"] != out.loc["2025-01-02 12:00", "p_L3"]  # NaN
    assert rep["phase_availability"]["L3"] < 1.0


def test_to_wide_rejects_missing_columns():
    with pytest.raises(ValueError):
        to_wide(pd.DataFrame({"AssetId": ["a"]}))


# ---------- leakage ----------
def test_split_is_chronological_and_disjoint():
    w = make_wide()
    tr, va, te, (t1, t2) = chronological_split(w)
    assert tr.index.max() < va.index.min() <= va.index.max() < te.index.min()
    assert len(tr) + len(va) + len(te) == len(w)


def test_context_depends_only_on_training_data():
    w = make_wide()
    tr, _, te, _ = chronological_split(w)
    a = fit_context(tr)
    te_changed = te.copy()
    te_changed["p_ALL"] *= 100          # must have no effect on anything fitted on train
    b = fit_context(tr)
    assert np.allclose(a.baseline.median_, b.baseline.median_)
    assert a.ref_share == b.ref_share and a.dz_scale == b.dz_scale


# ---------- injection integrity ----------
def test_unlabelled_windows_are_untouched_and_labelled_ones_changed():
    w = make_wide()
    _, _, te, _ = chronological_split(w)
    res = inject(te, seed=3, n_per_type=4, min_load_kw=0.5)
    assert len(res.events) > 0
    unl = (res.labels["is_injected"] == 0) & (res.labels["ambiguous"] == 0)
    for col in [f"p_{p}" for p in (*PHASES, "ALL")]:
        assert np.allclose(res.data.loc[unl, col], te.loc[unl, col]), f"{col} changed outside labelled/ambiguous windows"
    lab = res.labels["is_injected"] == 1
    diff = (res.data.loc[lab, "p_ALL"] - te.loc[lab, "p_ALL"]).abs()
    assert (diff >= 0.5 - 1e-9).all()


def test_injection_is_deterministic_and_total_stays_sum_of_phases():
    w = make_wide()
    _, _, te, _ = chronological_split(w)
    a, b = inject(te, seed=7, min_load_kw=0.5), inject(te, seed=7, min_load_kw=0.5)
    pd.testing.assert_frame_equal(a.data, b.data)
    s = a.data[[f"p_{p}" for p in PHASES]].sum(axis=1)
    assert np.allclose(s, a.data["p_ALL"])


# ---------- detectors / policy ----------
def test_dead_phase_triggers_phase_missing_rule_and_dq_routing():
    w = make_wide()
    tr, va, te, _ = chronological_split(w)
    ctx = fit_context(tr)
    rd = RuleDetector().fit(build_features(tr, ctx))
    broken = te.copy()
    broken.loc[broken.index[200:400], "p_L2"] = 0.0                  # reversed-CT style: clipped to zero
    broken["p_ALL"] = broken[[f"p_{p}" for p in PHASES]].sum(axis=1)
    f = build_features(broken, ctx)
    rules = rd.apply(f)
    loaded = f["active"].to_numpy()[200:400]
    assert rules["PHASE_MISSING"].iloc[200:400][loaded].mean() > 0.95
    # no false alarms on healthy part of the same segment
    assert rules["PHASE_MISSING"].iloc[:190].sum() == 0
    dec = decide(f, rules, pd.Series(False, index=f.index), pd.Series(np.nan, index=f.index))
    # PHASE_MISSING needs 2 consecutive windows, so the very first window of the fault is not yet flagged.
    seg = dec.iloc[201:400][loaded[1:]]
    assert (seg["verdict"] == "DATA_QUALITY_REVIEW").all()
    assert (seg["rules"].str.contains("PHASE_MISSING")).all()


def test_phase_stream_absent_is_flagged_even_when_idle():
    w = make_wide()
    tr, _, te, _ = chronological_split(w)
    ctx = fit_context(tr)
    t2 = te.copy()
    t2["p_L3"] = np.nan
    f = build_features(t2, ctx)
    assert f["missing_L3"].all()


def test_policy_never_assigns_urgent_without_a_facility_policy():
    w = make_wide()
    res = run_synthetic_benchmark(w, "x", n_per_type=3)
    assert "URGENT_ASSESSMENT" not in set(res.decisions["verdict"])
    assert res.audit_log["human_review_required"].all()
    assert (res.audit_log["reviewer_outcome"] == "").all()
    assert not any("injected" in c or "synthetic" in c for c in res.audit_log.columns)


def test_low_reliability_goes_to_data_quality_review():
    w = make_wide()
    tr, va, te, _ = chronological_split(w)
    ctx = fit_context(tr)
    te = te.copy()
    te.loc[te.index[50], "rel_cov"] = 40.0
    f = build_features(te, ctx)
    rd = RuleDetector().fit(build_features(tr, ctx))
    dec = decide(f, rd.apply(f), pd.Series(True, index=f.index), pd.Series(1.0, index=f.index))
    assert dec["verdict"].iloc[50] == "DATA_QUALITY_REVIEW"


# ---------- evaluation ----------
def test_group_alert_runs_merges_small_gaps():
    f = np.zeros(30, bool)
    f[[2, 3, 5, 12, 13, 20]] = True
    assert group_alert_runs(f, 2) == [(2, 5), (12, 13), (20, 20)]
    assert group_alert_runs(np.zeros(4, bool)) == []


def test_point_and_event_metrics_basic():
    idx = pd.date_range("2025-01-01", periods=100, freq="15min", tz="UTC")
    y = pd.Series(0, index=idx)
    y.iloc[10:20] = 1
    flag = pd.Series(False, index=idx)
    flag.iloc[12:15] = True
    flag.iloc[60:62] = True
    m = point_metrics(y, flag)
    assert m["tp"] == 3 and m["fp"] == 2 and m["fn"] == 7
    events = pd.DataFrame([{"event_id": 0, "anomaly_type": "t", "start": idx[10], "end": idx[19]}])
    labels = pd.DataFrame({"is_injected": y})
    per, summ = event_metrics(events, labels, flag)
    assert per["detected"].all() and per["delay_windows"].iloc[0] == 2
    assert summ["false_incidents"] == 1


def test_benchmark_runs_end_to_end_and_is_reproducible():
    w = make_wide(days=150)
    a = run_synthetic_benchmark(w, "x", seed=5, n_per_type=4)
    b = run_synthetic_benchmark(w, "x", seed=5, n_per_type=4)
    pd.testing.assert_frame_equal(a.metrics, b.metrics)
    assert a.event_summary.loc["rules OR iforest", "event_recall"] > 0.4
