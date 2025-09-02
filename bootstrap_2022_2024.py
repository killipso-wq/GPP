# bootstrap_2022_2024.py  (compatible with nfl-data-py 0.3.3)
import nfl_data_py as nfl
import pandas as pd
from pathlib import Path

YEARS = [2022, 2023, 2024]
DATA_DIR = Path("data_2022_2024")
DATA_DIR.mkdir(exist_ok=True)

def save(df: pd.DataFrame, name: str):
    out = DATA_DIR / f"{name}.parquet"
    try:
        out.unlink(missing_ok=True)
    except Exception:
        pass
    df.to_parquet(out, index=False)
    print(f"✓ wrote {out}  ({len(df):,} rows)")

print("Downloading and saving 2022–2024 data locally...")

# Signatures in 0.3.3
weekly = nfl.import_weekly_data(YEARS)
save(weekly, "weekly")

snaps = nfl.import_snap_counts(YEARS)
save(snaps, "snap_counts")

# Next Gen Stats may not exist in 0.3.3; try and skip if missing
try:
    ngs_pass = nfl.import_ngs_data("passing", YEARS)   # may raise AttributeError
    save(ngs_pass, "ngs_passing")
    ngs_rec  = nfl.import_ngs_data("receiving", YEARS)
    save(ngs_rec, "ngs_receiving")
    ngs_rush = nfl.import_ngs_data("rushing", YEARS)
    save(ngs_rush, "ngs_rushing")
except Exception as e:
    print("! NGS not available in your nfl-data-py version; skipping (ok for now).")
    print("  Details:", repr(e))

print("Done. Files saved to", DATA_DIR.resolve())
