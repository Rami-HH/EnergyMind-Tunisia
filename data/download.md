# Full Dataset Download & Citation

Full dataset: Flynn, C.; Murphy, T.; Walsh, J.; Riordan, D. Data 2026, 11(5), 101.
Paper: https://doi.org/10.3390/data11050101
Data (Zenodo): https://doi.org/10.5281/zenodo.19180972
License: CC BY 4.0.

## Instructions
1. Download the full industrial energy monitoring dataset from the Zenodo link above.
2. Extract the archive.
3. Pass the path of `data_csv/` to `src/prepare_dataset.py`:
   ```bash
   python src/prepare_dataset.py "path/to/data_csv" 2025-01-01 2025-12-31
   ```
