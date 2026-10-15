"""Feature engineering. Everything learned from data (profiles, reference shares, scales) is
fitted on the TRAINING segment only and then applied unchanged to later segments."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import PHASES

MAD_TO_SIGMA = 1.4826
MODEL_FEATURES = [
    "z_all", "dz", "log_load",
    "share_dev_L1", "share_dev_L2", "share_dev_L3", "n_phase_missing",
]


def _slot(index: pd.DatetimeIndex) -> np.ndarray:
    """Position in the week: day-of-week x 15-minute slot (0..671). Timestamps are UTC; in the
    dataset inspected, local time equals UTC, so no conversion is applied."""
    return (index.dayofweek * 96 + index.hour * 4 + index.minute // 15).to_numpy()


@dataclass
class SeasonalBaseline:
    """Robust weekly profile of total demand: median and MAD per weekly 15-min slot.

    Slot scales are floored with a fraction of the global scale because a few weeks of training
    give only a handful of samples per slot, which makes per-slot MAD unreliable.
    """
    scale_floor_frac: float = 0.25
    min_slot_samples: int = 3
    median_: np.ndarray = field(default=None, repr=False)
    scale_: np.ndarray = field(default=None, repr=False)

    def fit(self, series: pd.Series) -> "SeasonalBaseline":
        s = series.dropna()
        if len(s) < 7 * 96:
            raise ValueError("Need at least one week of clean training data to fit the baseline.")
        slots = _slot(s.index)
        g_med = float(s.median())
        # Floor the global scale relative to the load itself: for idle/burst processes the MAD is
        # ~0 and every normal burst would otherwise look like a huge outlier.
        g_scale = float(max(MAD_TO_SIGMA * (s - g_med).abs().median(), 0.1 * s.quantile(0.95), 1e-3))
        med = np.full(672, g_med)
        scale = np.full(672, g_scale)
        df = pd.DataFrame({"slot": slots, "v": s.to_numpy()})
        for slot, g in df.groupby("slot"):
            if len(g) >= self.min_slot_samples:
                m = float(g["v"].median())
                med[slot] = m
                scale[slot] = max(MAD_TO_SIGMA * float((g["v"] - m).abs().median()),
                                  self.scale_floor_frac * g_scale)
        self.median_, self.scale_, self.global_median_, self.global_scale_ = med, scale, g_med, g_scale
        return self

    def zscore(self, series: pd.Series) -> pd.Series:
        slots = _slot(series.index)
        return (series - self.median_[slots]) / self.scale_[slots]


@dataclass
class FeatureContext:
    """Everything fitted on the training segment."""
    baseline: SeasonalBaseline
    ref_share: dict          # phase -> median share of total load when the site is active
    ref_present: dict        # phase -> fraction of training windows where the phase stream exists
    min_load_kw: float       # below this, phase shares are meaningless
    dz_scale: float          # robust scale of 15-min changes in total demand


def fit_context(train: pd.DataFrame) -> FeatureContext:
    base = SeasonalBaseline().fit(train["p_ALL"])
    pos = train["p_ALL"].dropna()
    q95 = float(pos.quantile(0.95)) if len(pos) else 0.0
    min_load = float(max(0.05, 0.05 * q95))
    active = train["p_ALL"] > min_load
    ref_share, ref_present = {}, {}
    for ph in PHASES:
        col = train[f"p_{ph}"]
        ref_present[ph] = float(col[~train["gap"]].notna().mean()) if (~train["gap"]).any() else 0.0
        share = (col.fillna(0) / train["p_ALL"])[active]
        ref_share[ph] = float(share.median()) if active.any() and ref_present[ph] > 0 else 0.0
    d = train["p_ALL"].diff().dropna()
    dz_scale = float(max(MAD_TO_SIGMA * d.abs().median(), 0.05 * q95, 1e-3))
    return FeatureContext(base, ref_share, ref_present, min_load, dz_scale)


def build_features(df: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Row-aligned features for any segment. NaN where the window itself is missing."""
    f = pd.DataFrame(index=df.index)
    total = df["p_ALL"]
    f["z_all"] = ctx.baseline.zscore(total).clip(-500, 500)
    f["dz"] = (total.diff() / ctx.dz_scale).clip(-500, 500)
    f["log_load"] = np.log1p(total.clip(lower=0))

    active = total > ctx.min_load_kw
    n_missing = pd.Series(0, index=df.index, dtype=float)
    for ph in PHASES:
        col = df[f"p_{ph}"]
        share = col.fillna(0) / total.where(active)
        f[f"share_{ph}"] = share
        dev = (share - ctx.ref_share[ph]).where(active, 0.0)
        f[f"share_dev_{ph}"] = dev.where(~df["gap"])
        # A phase that normally exists and normally carries >=10% of the load, but is absent (NaN)
        # or ~0 (<2% of total) while the site is drawing load. This matches documented meter
        # faults: lost power tag (stream disappears) and reversed CT (reading clipped to zero).
        normally_present = ctx.ref_present[ph] >= 0.95
        absent = col.isna()
        dead = active & (ctx.ref_share[ph] >= 0.10) & (share < 0.02)
        vanished = normally_present & (absent | dead) & ~df["gap"]
        f[f"missing_{ph}"] = vanished
        n_missing += vanished.astype(float)
    f["n_phase_missing"] = n_missing.where(~df["gap"])
    f["active"] = active
    f["gap"] = df["gap"]
    f["rel_cov"] = df["rel_cov"]
    f["valid"] = f[MODEL_FEATURES].notna().all(axis=1)
    return f
