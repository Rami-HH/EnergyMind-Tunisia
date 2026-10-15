"""Synthetic anomaly injection with EVENT-level labels.

Injected anomalies are clearly labelled and live only in a copy of the test segment. They test
whether a detector recovers scenarios we created; they are not evidence of real-world fault
detection. Four scenario families, chosen to resemble documented metering problems:

  spike_drop  1-2 windows, all phases scaled x3 or x0.1 (sudden load event / dropout)
  gain_error  24-96 windows, one phase scaled x1.5 or x0.6 (e.g. CT ratio / scaling fault)
  drift       96-192 windows, one phase gain ramps away from 1 (sensor drift); only windows with
              >=15% deviation are labelled; the early ramp is marked "ambiguous" and not scored
  phase_loss  48-192 windows, one phase forced to 0 (e.g. reversed CT clipped to zero, lost tag)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import PHASES

TYPES = ("spike_drop", "gain_error", "drift", "phase_loss")


@dataclass
class InjectionResult:
    data: pd.DataFrame      # copy of the segment with anomalies applied
    labels: pd.DataFrame    # per-window: is_injected, event_id, anomaly_type
    events: pd.DataFrame    # per-event: event_id, anomaly_type, start, end, phase


def inject(seg: pd.DataFrame, seed: int, n_per_type: int = 6, min_load_kw: float = 0.05,
           buffer: int = 16) -> InjectionResult:
    rng = np.random.default_rng(seed)
    data = seg.copy()
    n = len(data)
    labels = pd.DataFrame({"is_injected": 0, "event_id": -1, "anomaly_type": "none", "ambiguous": 0},
                          index=data.index)
    occupied = np.zeros(n, dtype=bool)

    # Eligible: real, loaded windows with an intact recent past (so diff/baseline features exist).
    ok = (~data["gap"]) & (data["p_ALL"] > min_load_kw)
    ok = ok.rolling(8, min_periods=8).min().fillna(0).astype(bool).to_numpy()
    present = {ph: data[f"p_{ph}"].notna().to_numpy() for ph in PHASES}
    gap_arr = data["gap"].to_numpy()

    spec = {"spike_drop": (1, 2), "gain_error": (24, 96), "drift": (96, 192), "phase_loss": (48, 192)}
    events = []
    eid = 0

    def place(dur):
        for _ in range(400):
            s = int(rng.integers(8, max(9, n - dur - 1)))
            lo, hi = max(0, s - buffer), min(n, s + dur + buffer)
            if s + dur < n and ok[s] and not gap_arr[s:s + dur].any() and not occupied[lo:hi].any():
                occupied[lo:hi] = True
                return s
        return None

    p_all = data["p_ALL"].to_numpy().copy()
    ph_arr = {ph: data[f"p_{ph}"].to_numpy().copy() for ph in PHASES}

    for typ in TYPES:
        lo_d, hi_d = spec[typ]
        for _ in range(n_per_type):
            dur = int(rng.integers(lo_d, hi_d + 1))
            s = place(dur)
            if s is None:
                continue
            e = s + dur
            sl = slice(s, e)
            choices = [ph for ph in PHASES if present[ph][sl].all()]
            ph_hit = ""
            lab_mask = np.ones(dur, dtype=bool)
            changes: dict[str, np.ndarray] = {}     # phase -> new values (applied only if accepted)
            if typ == "spike_drop":
                g = float(rng.choice([3.0, 0.1]))
                changes = {ph: ph_arr[ph][sl] * g for ph in choices}
                ph_hit = "ALL"
            else:
                if not choices:
                    continue
                ph = str(rng.choice(choices))
                ph_hit = ph
                orig = ph_arr[ph][sl]
                if typ == "gain_error":
                    gain = np.full(dur, float(rng.choice([1.5, 0.6])))
                elif typ == "drift":
                    final = float(rng.choice([1.6, 0.5]))
                    gain = np.linspace(1.0, final, dur)
                    lab_mask = np.abs(gain - 1.0) >= 0.15
                else:  # phase_loss
                    gain = np.zeros(dur)
                changes = {ph: orig * gain}
            delta = sum((new - ph_arr[ph][sl] for ph, new in changes.items()), np.zeros(dur))
            # Label only windows where the injected change is material (>= min_load_kw): a "phase
            # loss" while the phase carries ~0 kW changes nothing and must not count as an event.
            lab_mask = lab_mask & (np.abs(delta) >= min_load_kw)
            if lab_mask.sum() < (1 if typ == "spike_drop" else 3):
                continue                              # rejected BEFORE touching the data
            for ph, new in changes.items():
                ph_arr[ph][sl] = new
            p_all[sl] += delta
            # Modified but not labelled (e.g. early part of a drift ramp): neither a clean negative
            # nor a clear positive, so it is marked "ambiguous" and excluded from scoring.
            amb = np.arange(s, e)[(~lab_mask) & (np.abs(delta) > 1e-9)]
            labels.iloc[amb, labels.columns.get_loc("ambiguous")] = 1
            idx = np.arange(s, e)[lab_mask]
            labels.iloc[idx, labels.columns.get_loc("is_injected")] = 1
            labels.iloc[idx, labels.columns.get_loc("event_id")] = eid
            labels.iloc[idx, labels.columns.get_loc("anomaly_type")] = typ
            events.append({"event_id": eid, "anomaly_type": typ, "phase": ph_hit,
                           "start": data.index[idx[0]], "end": data.index[idx[-1]],
                           "n_windows": len(idx)})
            eid += 1

    data["p_ALL"] = p_all
    for ph in PHASES:
        data[f"p_{ph}"] = ph_arr[ph]
    ev = pd.DataFrame(events, columns=["event_id", "anomaly_type", "phase", "start", "end", "n_windows"])
    return InjectionResult(data=data, labels=labels, events=ev)
