"""Load a slice of the Flynn et al. industrial 15-min dataset (partitioned CSV).

Folder layout:  data_csv/asset_id=<id>/dt_utc=<YYYY-MM-DD>/part-<YYYY-MM-DD>.csv

Usage:
    from load_data import load_assets, to_wide
    df = load_assets("path/to/data_csv", ["mix_a", "mix_b"], "2025-03-01", "2025-03-31")
    energy = to_wide(df, phase="ALL")          # rows = time, columns = assets
"""
from pathlib import Path
import pandas as pd

KEEP = ["AssetId", "Phase", "window_start_utc", "Energy_kWh_15m", "Demand_kW",
        "AvgPower_kW_15m", "DataCoveragePct", "ReliableCoveragePct", "IsReliableWindow"]


def load_assets(root, assets, start=None, end=None):
    """Read only the requested assets and UTC dates (inclusive, 'YYYY-MM-DD')."""
    root = Path(root)
    frames = []
    for a in assets:
        for day_dir in sorted((root / f"asset_id={a}").glob("dt_utc=*")):
            day = day_dir.name.split("=")[1]
            if (start and day < start) or (end and day > end):
                continue
            for f in day_dir.glob("*.csv"):
                frames.append(pd.read_csv(f, usecols=KEEP))
    if not frames:
        raise FileNotFoundError("No files found: check root path, asset ids and dates")
    df = pd.concat(frames, ignore_index=True)
    df["window_start_utc"] = pd.to_datetime(df["window_start_utc"])
    return df.sort_values(["AssetId", "Phase", "window_start_utc"]).reset_index(drop=True)


def to_wide(df, phase="ALL", value="Energy_kWh_15m"):
    """Pivot to one column per asset (handy for neighbour comparison)."""
    return (df[df.Phase == phase]
            .pivot(index="window_start_utc", columns="AssetId", values=value))


def consistency_checks(df):
    """Internal-consistency features that need no extra data.
    - Demand_kW should equal 4 * Energy_kWh_15m
    - ALL energy should equal L1 + L2 + L3
    """
    out = df.copy()
    out["demand_err"] = (out.Demand_kW - 4 * out.Energy_kWh_15m).abs()
    ph = (df[df.Phase != "ALL"].groupby(["AssetId", "window_start_utc"]).Energy_kWh_15m.sum()
          .rename("sum_phases").reset_index())
    allp = df[df.Phase == "ALL"].merge(ph, on=["AssetId", "window_start_utc"], how="left")
    allp["phase_sum_err"] = (allp.Energy_kWh_15m - allp.sum_phases).abs()
    return out, allp


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data_csv"
    d = load_assets(root, ["ahu_a"], "2025-01-01", "2025-01-07")
    print(d.head())
    print(to_wide(d).describe())
