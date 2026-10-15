"""Loading and validating the industrial 15-minute energy dataset (CC BY 4.0, C. Flynn / MTU)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PHASES = ("L1", "L2", "L3")
FREQ = "15min"
REQUIRED_COLUMNS = (
    "AssetId", "Phase", "window_start_utc", "Demand_kW",
    "ReliableCoveragePct", "IsReliableWindow",
)


def check_dataset_root(root: str | Path) -> Path:
    root = Path(root)
    if not (root / "data_csv").exists() and not (root / "data_parquet").exists():
        raise FileNotFoundError(
            f"No data_csv/ or data_parquet/ under {root}. "
            "Point DATA_ROOT at the extracted dataset folder."
        )
    return root


def list_assets(root: str | Path) -> list[str]:
    root = check_dataset_root(root)
    for sub in ("data_parquet", "data_csv"):
        if (root / sub).exists():
            return sorted(p.name.split("=", 1)[1] for p in (root / sub).glob("asset_id=*"))
    return []


def load_asset_long(root: str | Path, asset_id: str, prefer: str = "parquet") -> pd.DataFrame:
    """Read every day-partition of one asset (long format: one row per phase per window).

    Parquet is the default (needs pyarrow, listed in requirements.txt). Pass prefer="csv" to read
    data_csv/ instead. Falls back to the other format if one is unavailable.
    """
    root = check_dataset_root(root)
    order = ("parquet", "csv") if prefer == "parquet" else ("csv", "parquet")
    for kind in order:
        base = root / f"data_{kind}" / f"asset_id={asset_id}"
        files = sorted(base.glob(f"dt_utc=*/part-*.{kind}"))
        if not files:
            continue
        read = pd.read_parquet if kind == "parquet" else pd.read_csv
        try:
            return pd.concat((read(f) for f in files), ignore_index=True)
        except ImportError:  # parquet engine missing -> try the other format
            continue
    raise FileNotFoundError(f"No readable partitions for asset '{asset_id}' under {root}")


def to_wide(long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pivot to ONE row per 15-min window on a regular UTC grid.

    The raw files store L1, L2, L3 *and* ALL as separate rows. Treating them as a single
    time series (as the Phase 1 starter did) mixes four different signals, so we pivot first.

    Returns (wide, report). Columns of `wide`:
      p_L1, p_L2, p_L3  phase demand in kW (NaN = that phase stream is absent for the window)
      p_ALL             total demand in kW (NaN = whole window missing from the file)
      rel_cov           ReliableCoveragePct of the ALL stream
      gap               True where the window is absent from the file
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(long.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = long.copy()
    df["t"] = pd.to_datetime(df["window_start_utc"], utc=True, errors="coerce")
    n_bad_ts = int(df["t"].isna().sum())
    df = df.dropna(subset=["t"])
    n_dup = int(df.duplicated(["AssetId", "Phase", "t"]).sum())
    df = df.drop_duplicates(["AssetId", "Phase", "t"], keep="last")
    df["Demand_kW"] = pd.to_numeric(df["Demand_kW"], errors="coerce")
    df.loc[~np.isfinite(df["Demand_kW"]), "Demand_kW"] = np.nan

    p = df.pivot(index="t", columns="Phase", values="Demand_kW")
    wide = pd.DataFrame(index=p.index)
    for ph in (*PHASES, "ALL"):
        wide[f"p_{ph}"] = p[ph] if ph in p else np.nan
    rel = df[df["Phase"] == "ALL"].set_index("t")["ReliableCoveragePct"]
    wide["rel_cov"] = pd.to_numeric(rel, errors="coerce")

    grid = pd.date_range(wide.index.min(), wide.index.max(), freq=FREQ, tz="UTC")
    wide = wide.reindex(grid).sort_index()
    wide.index.name = "window_start_utc"
    wide["gap"] = wide["p_ALL"].isna()

    report = {
        "rows_long": len(long),
        "windows_on_grid": len(wide),
        "gap_windows": int(wide["gap"].sum()),
        "duplicate_rows_dropped": n_dup,
        "unparseable_timestamps": n_bad_ts,
        "negative_demand_rows": int((df["Demand_kW"] < 0).sum()),
        "first_window": wide.index.min(),
        "last_window": wide.index.max(),
        "phase_availability": {ph: float(wide[f"p_{ph}"].notna().mean()) for ph in PHASES},
        "unreliable_share": float((wide["rel_cov"] < 80).mean()),
    }
    return wide, report


def load_quality_log(root: str | Path) -> pd.DataFrame:
    """Meter quality log. Dates are DD/MM/YYYY HH:MM. The log does not state a timezone, so we
    treat them as UTC (local and UTC timestamps are identical in the files inspected) and say
    so wherever the log is used."""
    path = Path(root) / "metadata" / "meter_data_quality_log.csv"
    log = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("issue_start", "issue_end", "asset_stream_start", "asset_stream_end"):
        log[col] = pd.to_datetime(log[col], format="%d/%m/%Y %H:%M", utc=True, errors="coerce")
    log["asset_id"] = log["asset_id"].astype(str).str.lower()
    return log


def issue_mask(index: pd.DatetimeIndex, log: pd.DataFrame, asset_id: str) -> pd.Series:
    """True for windows inside any logged issue interval of this asset."""
    m = pd.Series(False, index=index)
    for _, r in log[log["asset_id"] == asset_id.lower()].iterrows():
        if pd.notna(r["issue_start"]) and pd.notna(r["issue_end"]):
            m |= (index >= r["issue_start"]) & (index <= r["issue_end"])
    return m
