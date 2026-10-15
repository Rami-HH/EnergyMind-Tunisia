"""Run the Phase 1 benchmark across assets.

    python scripts/run_benchmark.py --out results/

Writes:
  results/benchmark_by_asset.csv     event-level and window-level numbers per asset and detector
  results/benchmark_summary.csv      macro-average across assets
  results/real_fault_case_studies.csv
  results/audit_log_<asset>.csv      one audit row per incident (ahu_a / mix_b case studies)

Assets with logged issues are excluded from the synthetic benchmark (their logged-issue windows
are blanked out), so the benchmark's "clean" data really is clean.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energymind.data import issue_mask, list_assets, load_asset_long, load_quality_log, to_wide  # noqa: E402
from energymind.pipeline import run_real_fault_case_study, run_synthetic_benchmark  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "dataset"),
                    help="Dataset folder (default: ./dataset shipped with this project)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--assets", nargs="*", help="Subset of asset ids (default: all)")
    ap.add_argument("--min-events", type=int, default=8,
                    help="Skip assets where fewer synthetic events could be injected")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)
    log = load_quality_log(root)
    assets = args.assets or list_assets(root)

    rows, skipped, cases = [], [], []
    for a in assets:
        try:
            wide, rep = to_wide(load_asset_long(root, a, prefer="parquet"))
        except Exception as e:
            skipped.append({"asset": a, "reason": f"load failed: {type(e).__name__}: {e}"})
            continue

        # Real-fault case study (independent of the synthetic benchmark).
        if (log["asset_id"] == a.lower()).any():
            try:
                cs = run_real_fault_case_study(wide, log, a, seed=args.seed)
                s = cs.summary.reset_index()
                s.insert(0, "asset", a)
                s["issue_start"] = cs.issue_start
                cases.append(s)
                cs.audit_log.to_csv(out / f"audit_log_{a}.csv", index=False)
                print(f"[case study] {a}")
            except ValueError:
                pass  # no clean baseline before the first issue

        try:
            excl = issue_mask(wide.index, log, a)
            res = run_synthetic_benchmark(wide, a, exclude=excl if excl.any() else None, seed=args.seed)
        except Exception as e:  # keep going; report why an asset was skipped
            skipped.append({"asset": a, "reason": f"{type(e).__name__}: {e}"})
            continue
        if len(res.events) < args.min_events:
            skipped.append({"asset": a, "reason": f"only {len(res.events)} injectable events"})
            continue
        for det in res.metrics.index:
            rows.append({"asset": a, "detector": det, "n_events": len(res.events),
                         "window_precision": res.metrics.loc[det, "precision"],
                         "window_recall": res.metrics.loc[det, "recall"],
                         "event_recall": res.event_summary.loc[det, "event_recall"],
                         "false_incidents_per_day": res.event_summary.loc[det, "false_incidents_per_day"]})
        print(f"[ok] {a}: {len(res.events)} events")

    by_asset = pd.DataFrame(rows)
    by_asset.to_csv(out / "benchmark_by_asset.csv", index=False)
    if len(by_asset):
        summ = (by_asset.groupby("detector")[["window_precision", "window_recall", "event_recall",
                                              "false_incidents_per_day"]].mean().round(3))
        summ["assets"] = by_asset.groupby("detector")["asset"].nunique()
        summ.to_csv(out / "benchmark_summary.csv")
        print("\nMacro-average over assets:\n", summ.to_string())
    if cases:
        pd.concat(cases).to_csv(out / "real_fault_case_studies.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(out / "skipped_assets.csv", index=False)
        print(f"\nSkipped {len(skipped)} assets (see skipped_assets.csv)")


if __name__ == "__main__":
    main()
