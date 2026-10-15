"""Two detectors with complementary roles.

RuleDetector      transparent, each rule has a name an operator can read.
IForestDetector   catches unusual *combinations* of features that no single rule covers.

Thresholds are never hand-picked on the test set:
  * rule thresholds come from quantiles of the clean TRAINING features (target false-alert rate),
  * the Isolation Forest alert threshold is the same kind of quantile taken on a clean
    VALIDATION segment the model has not seen (in-sample scores are over-optimistic).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .data import PHASES
from .features import MODEL_FEATURES

RULE_NAMES = ["PHASE_MISSING", "PHASE_SHARE_SHIFT", "SEASONAL_DEVIATION", "ABRUPT_CHANGE"]

RULE_TEXT = {
    "PHASE_MISSING": "A phase stream that normally exists is absent or ~0 while the site draws load",
    "PHASE_SHARE_SHIFT": "Load split between phases differs persistently from the training baseline",
    "SEASONAL_DEVIATION": "Total demand persistently outside the usual range for this time of week",
    "ABRUPT_CHANGE": "Sudden jump or drop in total demand between consecutive windows",
}


def _persist(flag: pd.Series, k: int) -> pd.Series:
    """True only where `flag` has been True for k consecutive windows (NaN counts as False)."""
    if k <= 1:
        return flag.fillna(False).astype(bool)
    return flag.fillna(False).astype(float).rolling(k, min_periods=k).min().fillna(0).astype(bool)


@dataclass
class RuleDetector:
    fa_rate: float = 0.005        # target false-alert rate per rule on clean training data
    persist: int = 4              # windows (= 1 hour) a level/share rule must hold
    z_floor: float = 4.0
    dz_floor: float = 6.0
    share_floor: float = 0.10
    thresholds_: dict = field(default_factory=dict)

    def fit(self, feat_train: pd.DataFrame) -> "RuleDetector":
        t = feat_train[feat_train["valid"]]
        if len(t) < 500:
            raise ValueError("Too little valid training data to calibrate rule thresholds.")
        sh = t[[f"share_dev_{p}" for p in PHASES]].abs().max(axis=1)
        q = 1 - self.fa_rate
        self.thresholds_ = {
            "z": max(float(t["z_all"].abs().quantile(q)), self.z_floor),
            "dz": max(float(t["dz"].abs().quantile(q)), self.dz_floor),
            "share": max(float(sh.quantile(q)), self.share_floor),
        }
        return self

    def apply(self, feat: pd.DataFrame) -> pd.DataFrame:
        th = self.thresholds_
        out = pd.DataFrame(index=feat.index)
        miss = feat[[f"missing_{p}" for p in PHASES]].any(axis=1)
        # also: phase present but ~0 while the site is loaded and the phase normally carries load
        share_cols = [f"share_dev_{p}" for p in PHASES]
        out["PHASE_MISSING"] = _persist(miss, 2)
        out["PHASE_SHARE_SHIFT"] = _persist(feat[share_cols].abs().max(axis=1) > th["share"], self.persist)
        out["SEASONAL_DEVIATION"] = _persist(feat["z_all"].abs() > th["z"], self.persist)
        out["ABRUPT_CHANGE"] = (feat["dz"].abs() > th["dz"]).fillna(False)
        out = out.where(feat["valid"] | out["PHASE_MISSING"], False)
        return out.astype(bool)


@dataclass
class IForestDetector:
    n_estimators: int = 300
    fa_rate: float = 0.01
    random_state: int = 42
    threshold_: float = None
    model_: IsolationForest = field(default=None, repr=False)

    def fit(self, feat_train: pd.DataFrame, feat_val: pd.DataFrame) -> "IForestDetector":
        tr = feat_train.loc[feat_train["valid"], MODEL_FEATURES]
        va = feat_val.loc[feat_val["valid"], MODEL_FEATURES]
        if len(tr) < 500 or len(va) < 200:
            raise ValueError("Too little valid data to fit/calibrate the Isolation Forest.")
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators, contamination="auto",
            random_state=self.random_state, n_jobs=-1,
        ).fit(tr)
        val_scores = -self.model_.score_samples(va)
        self.threshold_ = float(np.quantile(val_scores, 1 - self.fa_rate))
        return self

    def score(self, feat: pd.DataFrame) -> pd.Series:
        s = pd.Series(np.nan, index=feat.index)
        v = feat["valid"]
        if v.any():
            s[v] = -self.model_.score_samples(feat.loc[v, MODEL_FEATURES])
        return s

    def flag(self, scores: pd.Series) -> pd.Series:
        return (scores > self.threshold_).fillna(False)
