"""Step 1: build a clean long-format table for the 4 mixers.

Usage:
    python src/prepare_dataset.py "C:/path/to/data_csv" 2025-01-01 2025-12-31
Output: data/clean_mixers.csv  (time, asset, E, L1, L2, L3, IsReliableWindow, DataCoveragePct)
"""
import sys
from pathlib import Path
import pandas as pd

# Add src to sys.path if run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_assets

ASSETS = ["mix_a", "mix_b", "mix_c", "mix_d"]


def build_clean(root, assets=ASSETS, start=None, end=None):
    df = load_assets(root, assets, start, end)
    w = df.pivot_table(index=["AssetId", "window_start_utc"], columns="Phase",
                       values="Energy_kWh_15m").reset_index().rename(columns={"ALL": "E"})
    rel = df[df.Phase == "ALL"][["AssetId", "window_start_utc", "IsReliableWindow", "DataCoveragePct"]]
    out = w.merge(rel, on=["AssetId", "window_start_utc"])
    out = out.rename(columns={"AssetId": "asset", "window_start_utc": "time"})
    return out[["time", "asset", "E", "L1", "L2", "L3", "IsReliableWindow", "DataCoveragePct"]] \
        .sort_values(["time", "asset"]).reset_index(drop=True)


if __name__ == "__main__":
    root, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    clean = build_clean(root, ASSETS, start, end)
    Path("data").mkdir(exist_ok=True)
    clean.to_csv("data/clean_mixers.csv", index=False)
    print(clean.shape, "->", "data/clean_mixers.csv")
    print(clean.groupby("asset").E.describe().round(3))
