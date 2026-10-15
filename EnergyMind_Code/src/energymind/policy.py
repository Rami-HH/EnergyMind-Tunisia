"""Decision policy and audit log, following GOVERNANCE_AND_SHARED_LEARNING.md.

The system only RECOMMENDS REVIEW. It never controls equipment. "URGENT" is never assigned
automatically: thresholds for high-impact events must be agreed with a qualified facility operator,
so a deployment must supply its own `urgent_policy`.

Instead of the starter's invented weighted "confidence score", we report an EVIDENCE-AGREEMENT
band (what independent evidence sources agree), which is honest about what it is:
    HIGH   rule(s) AND model agree
    MEDIUM two or more distinct rules agree, model silent
    LOW    a single rule OR the model alone
It is not a probability and is not calibrated.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import MODEL_VERSION
from .detectors import RULE_NAMES, RULE_TEXT
from .evaluate import group_alert_runs

LOW_RELIABLE_COVERAGE = 80.0

ACTIONS = {
    "DATA_QUALITY_REVIEW": "Verify measurement reliability and meter configuration (CT polarity, tags, ratios)",
    "OPERATOR_REVIEW": "Operator should review the evidence before any corrective action",
    "URGENT_ASSESSMENT": "Urgent human assessment per facility-specific policy",
}


def decide(feat: pd.DataFrame, rules: pd.DataFrame, model_flag: pd.Series, model_score: pd.Series,
           urgent_policy: Callable[[pd.DataFrame], pd.Series] | None = None) -> pd.DataFrame:
    """Per-window verdicts. Data-quality problems take precedence over behavioural alerts, because
    a model alert on unreliable data is not trustworthy evidence of an equipment event."""
    d = pd.DataFrame(index=feat.index)
    n_rules = rules[RULE_NAMES].sum(axis=1)
    mflag = model_flag.reindex(feat.index).fillna(False)

    # A vanished/dead phase is a meter-configuration symptom (governance: "meter configuration issue
    # indicated"), so it goes to data-quality review, with the rule still listed as evidence.
    dq = (feat["gap"] | feat["rel_cov"].lt(LOW_RELIABLE_COVERAGE) | feat["rel_cov"].isna()
          | rules["PHASE_MISSING"])
    behav = (n_rules > 0) | mflag

    verdict = np.where(dq, "DATA_QUALITY_REVIEW", np.where(behav, "OPERATOR_REVIEW", "ROUTINE"))
    d["verdict"] = verdict
    if urgent_policy is not None:
        urgent = urgent_policy(feat).reindex(feat.index).fillna(False) & (d["verdict"] != "ROUTINE")
        d.loc[urgent, "verdict"] = "URGENT_ASSESSMENT"

    d["agreement"] = np.select(
        [(n_rules > 0) & mflag, n_rules >= 2, behav], ["HIGH", "MEDIUM", "LOW"], default="")
    d["rules"] = rules[RULE_NAMES].apply(lambda r: ";".join(n for n in RULE_NAMES if r[n]), axis=1)
    d["model_flag"] = mflag
    d["model_score"] = model_score.reindex(feat.index)
    d["human_review_required"] = d["verdict"] != "ROUTINE"
    return d


def build_audit_log(asset_id: str, feat: pd.DataFrame, decisions: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """One audit row per INCIDENT (consecutive alert windows merged), not per 15-min window.
    Reviewer fields are left blank for the human to fill in - they are part of the audit trail."""
    alert = (decisions["verdict"] != "ROUTINE").to_numpy()
    rows = []
    for k, (s, e) in enumerate(group_alert_runs(alert, max_gap), start=1):
        seg = decisions.iloc[s:e + 1]
        seg = seg[seg["verdict"] != "ROUTINE"]
        rules = sorted({r for cell in seg["rules"] for r in cell.split(";") if r})
        order = ["URGENT_ASSESSMENT", "OPERATOR_REVIEW", "DATA_QUALITY_REVIEW"]
        verdict = next(v for v in order if (seg["verdict"] == v).any())
        band = next((b for b in ("HIGH", "MEDIUM", "LOW") if (seg["agreement"] == b).any()), "")
        rows.append({
            "event_id": f"EM-{asset_id}-{k:05d}",
            "asset_id": asset_id,
            "start_utc": decisions.index[s], "end_utc": decisions.index[e],
            "n_alert_windows": int(len(seg)),
            "rule_evidence": "; ".join(f"{r}: {RULE_TEXT[r]}" for r in rules),
            "model_flagged": bool(seg["model_flag"].any()),
            "peak_model_score": float(seg["model_score"].max()) if seg["model_score"].notna().any() else np.nan,
            "evidence_agreement": band,
            "verdict": verdict,
            "recommended_action": ACTIONS[verdict],
            "human_review_required": True,
            "model_version": MODEL_VERSION,
            "reviewer_outcome": "", "reviewed_at_utc": "",
        })
    return pd.DataFrame(rows)
