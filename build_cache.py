import os
import pandas as pd
import nfl_data_py as nfl

YEARS = [2022, 2023, 2024]
ROOT = os.path.join(os.path.dirname(__file__), "data_cache")
PARQ = os.path.join(ROOT, "parquet")
CSV  = os.path.join(ROOT, "csv")  # optional for inspection

os.makedirs(PARQ, exist_ok=True)
os.makedirs(CSV, exist_ok=True)

def save(df: pd.DataFrame, name: str, index=False):
    p = os.path.join(PARQ, f"{name}.parquet")
    c = os.path.join(CSV,  f"{name}.csv")
    df.to_parquet(p, index=index)
    df.to_csv(c, index=index)
    print(f"✓ Saved {name}: {len(df):,} rows")

def fetch(name: str, func, **kwargs):
    print(f"Downloading {name}…")
    df = func(**kwargs)
    save(df, name)
    return df

def main():
    print("=== Building local NFL cache (2022–2024) ===")
    weekly = fetch("weekly", nfl.import_weekly_data, years=YEARS)
    pbp    = fetch("pbp", nfl.import_pbp_data, years=YEARS, downcast=True, cache=False)
    rosters= fetch("rosters", nfl.import_seasonal_rosters, years=YEARS)
    snaps  = fetch("snap_counts", nfl.import_snap_counts, years=YEARS)
    ngs_recv = fetch("ngs_receiving", nfl.import_ngs_data, stat_type="receiving", years=YEARS)
    ngs_rush = fetch("ngs_rushing",   nfl.import_ngs_data, stat_type="rushing",   years=YEARS)
    ngs_pass = fetch("ngs_passing",   nfl.import_ngs_data, stat_type="passing",   years=YEARS)

    print("Computing team offense/defense summaries…")
    df = pbp[pbp["play_type"].isin(["pass","run"])].copy()
    for c in ["pass","rush","success"]:
        df[c] = df[c].fillna(0).astype(float)
    df["yards_gained"] = df["yards_gained"].fillna(0).astype(float)
    df["epa"] = df["epa"].fillna(0.0).astype(float)

    offense = (df.groupby(["season","posteam"])
        .agg(plays=("epa","size"),
             pass_plays=("pass","sum"),
             rush_plays=("rush","sum"),
             yards=("yards_gained","sum"),
             epa_per_play=("epa","mean"),
             success_rate=("success","mean"))
        .reset_index().rename(columns={"posteam":"team"}))

    defense = (df.groupby(["season","defteam"])
        .agg(plays=("epa","size"),
             yards_allowed=("yards_gained","sum"),
             epa_per_play_allowed=("epa","mean"),
             success_rate_allowed=("success","mean"))
        .reset_index().rename(columns={"defteam":"team"}))

    save(offense, "team_offense_summary")
    save(defense, "team_defense_summary")
    save(offense.merge(defense, on=["season","team"], how="outer"), "team_off_def_combined")

    print("All done. Local cache is ready.")

if __name__ == "__main__":
    main()
