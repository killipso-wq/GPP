# download_nfl_data.py
# Exports 2022–2024 weekly player stats, plus team offense/defense summaries.

import os
import pandas as pd
import nfl_data_py as nfl

YEARS = [2022, 2023, 2024]

EXPORT_WEEKLY = True
EXPORT_TEAM_OFF_DEF = True
EXPORT_PBP = False   # WARNING: huge CSV if True

OUTDIR = "out_csv"
os.makedirs(OUTDIR, exist_ok=True)

def export_weekly_player_stats(years):
    print("Downloading weekly player stats...")
    df = nfl.import_weekly_data(years)
    out_path = os.path.join(OUTDIR, f"weekly_player_stats_{years[0]}_{years[-1]}.csv")
    df.to_csv(out_path, index=False)
    print(f"✓ Wrote {len(df):,} rows -> {out_path}")

def export_team_offense_defense_summaries(years):
    print("Downloading play-by-play for summaries (can take a bit)...")
    pbp = nfl.import_pbp_data(years, downcast=True, cache=False)

    mask = pbp["play_type"].isin(["pass", "run"])
    cols = ["season", "posteam", "defteam", "yards_gained", "epa", "success", "pass", "rush"]
    df = pbp.loc[mask, cols].copy()

    for c in ["pass", "rush", "success"]:
        df[c] = df[c].fillna(0).astype(float)
    df["yards_gained"] = df["yards_gained"].fillna(0).astype(float)
    df["epa"] = df["epa"].fillna(0.0).astype(float)

    offense = (
        df.groupby(["season", "posteam"])
          .agg(
              plays=("epa", "size"),
              pass_plays=("pass", "sum"),
              rush_plays=("rush", "sum"),
              yards=("yards_gained", "sum"),
              epa_per_play=("epa", "mean"),
              success_rate=("success", "mean"),
          )
          .reset_index()
          .rename(columns={"posteam": "team"})
    )

    defense = (
        df.groupby(["season", "defteam"])
          .agg(
              plays=("epa", "size"),
              yards_allowed=("yards_gained", "sum"),
              epa_per_play_allowed=("epa", "mean"),
              success_rate_allowed=("success", "mean"),
          )
          .reset_index()
          .rename(columns={"defteam": "team"})
    )

    off_path = os.path.join(OUTDIR, f"team_offense_summary_{years[0]}_{years[-1]}.csv")
    def_path = os.path.join(OUTDIR, f"team_defense_summary_{years[0]}_{years[-1]}.csv")
    offense.to_csv(off_path, index=False)
    defense.to_csv(def_path, index=False)

    print(f"✓ Wrote offense summary ({len(offense):,} rows) -> {off_path}")
    print(f"✓ Wrote defense summary ({len(defense):,} rows) -> {def_path}")

def export_raw_pbp(years):
    print("Downloading full play-by-play (very large)...")
    pbp = nfl.import_pbp_data(years, downcast=True, cache=True)
    out_path_csv = os.path.join(OUTDIR, f"pbp_{years[0]}_{years[-1]}.csv")
    pbp.to_csv(out_path_csv, index=False)
    print(f"✓ Wrote {len(pbp):,} rows -> {out_path_csv}")

if __name__ == "__main__":
    if EXPORT_WEEKLY:
        export_weekly_player_stats(YEARS)
    if EXPORT_TEAM_OFF_DEF:
        export_team_offense_defense_summaries(YEARS)
    if EXPORT_PBP:
        export_raw_pbp(YEARS)
    print("All done.")
