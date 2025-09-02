# build_features_2022_2024.py (robust NGS handling for nfl-data-py 0.3.3)
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data_2022_2024")
OUT = DATA_DIR  # write outputs here

def readp(name):
    p = DATA_DIR / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run bootstrap_2022_2024.py first.")
    return pd.read_parquet(p)

def pick(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def first_col(df, *names):
    for n in names:
        if n and n in df.columns:
            return n
    return None

# Load base tables
weekly = readp("weekly")
snaps  = readp("snap_counts")

# Try NGS (optional)
ngs_receiving_p = (DATA_DIR/"ngs_receiving.parquet")
ngs_passing_p   = (DATA_DIR/"ngs_passing.parquet")
ngs_rushing_p   = (DATA_DIR/"ngs_rushing.parquet")
ngs_rec = pd.read_parquet(ngs_receiving_p) if ngs_receiving_p.exists() else None
ngs_pas = pd.read_parquet(ngs_passing_p)   if ngs_passing_p.exists()   else None
ngs_rus = pd.read_parquet(ngs_rushing_p)   if ngs_rushing_p.exists()   else None

# --- Column mapping for weekly (robust) ---
name_col   = pick(weekly, ["player_name","player","full_name"], "player_name")
id_col     = pick(weekly, ["player_id","gsis_id","pfr_id"], "player_id")
team_col   = pick(weekly, ["team","recent_team","posteam"], "team")
opp_col    = pick(weekly, ["opponent_team","opponent","opp","defteam"], None)
pos_col    = pick(weekly, ["position","pos"], "position")
season_col = pick(weekly, ["season","year"], "season")
week_col   = pick(weekly, ["week","game_week"], "week")

tgt_col    = pick(weekly, ["targets","target"], None)
rec_col    = pick(weekly, ["receptions","rec"], None)
recyd_col  = pick(weekly, ["receiving_yards","rec_yards","yards_receiving"], None)
rectd_col  = pick(weekly, ["receiving_tds","rec_tds"], None)

rushatt_col= pick(weekly, ["rushing_attempts","rush_attempts","carries"], None)
rushyd_col = pick(weekly, ["rushing_yards","rush_yards","yards_rushing"], None)
rushtd_col = pick(weekly, ["rushing_tds","rush_tds"], None)

passatt_col= pick(weekly, ["attempts","pass_attempts"], None)

req = [name_col,id_col,team_col,season_col,week_col]
if any(r is None or r not in weekly.columns for r in req):
    missing = [r for r in req if r is None or r not in weekly.columns]
    raise ValueError(f"Weekly dataset missing required columns: {missing}. Share a sample and I will map.")

# Normalize a working weekly frame
keep = [id_col,name_col,team_col,season_col,week_col] + [c for c in [opp_col,pos_col,tgt_col,rec_col,recyd_col,rectd_col,rushatt_col,rushyd_col,rushtd_col,passatt_col] if c]
wk = weekly[keep].copy()
wk.rename(columns={
    id_col:"player_id", name_col:"player_name", team_col:"team",
    season_col:"season", week_col:"week"
}, inplace=True)
if opp_col: wk.rename(columns={opp_col:"opp"}, inplace=True)
if pos_col: wk.rename(columns={pos_col:"pos"}, inplace=True)
if tgt_col: wk.rename(columns={tgt_col:"targets"}, inplace=True)
if rec_col: wk.rename(columns={rec_col:"receptions"}, inplace=True)
if recyd_col: wk.rename(columns={recyd_col:"receiving_yards"}, inplace=True)
if rectd_col: wk.rename(columns={rectd_col:"receiving_tds"}, inplace=True)
if rushatt_col: wk.rename(columns={rushatt_col:"rushing_attempts"}, inplace=True)
if rushyd_col: wk.rename(columns={rushyd_col:"rushing_yards"}, inplace=True)
if rushtd_col: wk.rename(columns={rushtd_col:"rushing_tds"}, inplace=True)
if passatt_col: wk.rename(columns={passatt_col:"pass_attempts"}, inplace=True)

# Clean types
for c in ["targets","receptions","receiving_yards","receiving_tds",
          "rushing_attempts","rushing_yards","rushing_tds","pass_attempts"]:
    if c in wk.columns:
        wk[c] = pd.to_numeric(wk[c], errors="coerce").fillna(0)

wk["team"] = wk["team"].astype(str).str.upper().str.strip()
if "opp" in wk.columns:
    wk["opp"] = wk["opp"].astype(str).str.upper().str.strip()
if "pos" in wk.columns:
    wk["pos"] = wk["pos"].astype(str).str.upper().str.strip()

# Team totals by week
team_week = wk.groupby(["team","season","week"], as_index=False).agg(
    team_targets       = ("targets","sum") if "targets" in wk.columns else ("player_id","size"),
    team_rush_attempts = ("rushing_attempts","sum") if "rushing_attempts" in wk.columns else ("player_id","size"),
    team_pass_att_sum  = ("pass_attempts","sum") if "pass_attempts" in wk.columns else ("player_id","size")
)
team_week["team_plays"] = 0
if "targets" in wk.columns and "rushing_attempts" in wk.columns:
    team_week["team_plays"] = team_week["team_targets"].fillna(0) + team_week["team_rush_attempts"].fillna(0)
team_week.loc[team_week["team_plays"] < 40, "team_plays"] = np.nan
team_week["team_pass_rate"] = np.where(team_week["team_plays"]>0,
                                       team_week["team_targets"]/team_week["team_plays"], np.nan)

wk = wk.merge(team_week, on=["team","season","week"], how="left")

# Shares & efficiencies
wk["target_share"] = np.where((wk.get("team_targets",0)>0) & ("targets" in wk.columns),
                              wk["targets"]/wk["team_targets"], np.nan)
wk["rush_share"]   = np.where((wk.get("team_rush_attempts",0)>0) & ("rushing_attempts" in wk.columns),
                              wk["rushing_attempts"]/wk["team_rush_attempts"], np.nan)

wk["YPT"] = np.where((wk.get("targets",0)>0) & ("receiving_yards" in wk.columns),
                     wk["receiving_yards"]/wk["targets"], np.nan)
wk["YPC"] = np.where((wk.get("rushing_attempts",0)>0) & ("rushing_yards" in wk.columns),
                     wk["rushing_yards"]/wk["rushing_attempts"], np.nan)

wk["rec_tdrate"]  = np.where((wk.get("targets",0)>0) & ("receiving_tds" in wk.columns),
                             wk["receiving_tds"]/wk["targets"], np.nan)
wk["rush_tdrate"] = np.where((wk.get("rushing_attempts",0)>0) & ("rushing_tds" in wk.columns),
                             wk["rushing_tds"]/wk["rushing_attempts"], np.nan)

# Optional NGS receiving enrichments — robust column detection
if ngs_rec is not None:
    # Try a wide set of possible column names seen in various dumps
    rec_id  = first_col(ngs_rec, "player_id","gsis_id","nfl_id","nflId",
                        "receiver_gsis_id","receiver_id","gsis_player_id","player_gsis_id")
    rec_season = first_col(ngs_rec, "season","year","Season")
    rec_week   = first_col(ngs_rec, "week","game_week","Week","wk")
    routes_col = first_col(ngs_rec, "routes","routes_run","routes_run_total","Routes","route_runs")
    sep_col    = first_col(ngs_rec, "avg_sep","average_separation","avg_separation",
                           "avg_separation_yards","separation","avgSeparation")
    yacoe_col  = first_col(ngs_rec, "yac_over_expected","yac_oe","yacoe","yacOE","yac_oe_per_reception")

    if not rec_id:
        print("! NGS receiving present but no player-ID column found — skipping NGS enrich (ok).")
        print("  NGS columns sample:", list(ngs_rec.columns)[:25])
    else:
        cols = [x for x in [rec_id, rec_season, rec_week, routes_col, sep_col, yacoe_col] if x]
        # Keep only those that truly exist
        cols = [c for c in cols if c in ngs_rec.columns]
        ngr = ngs_rec[cols].copy()

        rename_map = {}
        rename_map[rec_id] = "player_id"
        if rec_season: rename_map[rec_season] = "season"
        if rec_week:   rename_map[rec_week]   = "week"
        if routes_col: rename_map[routes_col] = "routes"
        if sep_col:    rename_map[sep_col]    = "avg_sep"
        if yacoe_col:  rename_map[yacoe_col]  = "yac_oe"

        ngr.rename(columns=rename_map, inplace=True)

        for c in ["routes","avg_sep","yac_oe"]:
            if c in ngr.columns:
                ngr[c] = pd.to_numeric(ngr[c], errors="coerce")

        # Only merge if we have ID + season + week to align properly
        merge_keys = ["player_id"]
        if "season" in ngr.columns: merge_keys.append("season")
        if "week"   in ngr.columns: merge_keys.append("week")

        if set(merge_keys) <= set(ngr.columns):
            wk = wk.merge(ngr, on=merge_keys, how="left")
            if "routes" in wk.columns and "team_pass_att_sum" in wk.columns:
                wk["route_participation"] = np.where(wk["team_pass_att_sum"]>0,
                                                     wk["routes"]/wk["team_pass_att_sum"], np.nan)
        else:
            print("! NGS receiving lacks season/week to align — skipping merge (ok).")

# Aggregate → PlayerUsage (player, team, pos)
gb_cols = ["player_id","player_name","team"] + (["pos"] if "pos" in wk.columns else [])
gb = wk.groupby(gb_cols, as_index=False)

def safe_mean(s):
    s = s.dropna()
    return s.mean() if len(s) else np.nan

def safe_sd(s):
    s = s.dropna()
    return s.std(ddof=0) if len(s) else np.nan

usage = gb.agg(
    target_share_mean = ("target_share", safe_mean) if "target_share" in wk.columns else ("team_plays", safe_mean),
    target_share_sd   = ("target_share", safe_sd)   if "target_share" in wk.columns else ("team_plays", safe_sd),
    rush_share_mean   = ("rush_share", safe_mean)   if "rush_share" in wk.columns   else ("team_plays", safe_mean),
    rush_share_sd     = ("rush_share", safe_sd)     if "rush_share" in wk.columns   else ("team_plays", safe_sd),

    ypt_mean = ("YPT", safe_mean) if "YPT" in wk.columns else ("team_plays", safe_mean),
    ypt_sd   = ("YPT", safe_sd)   if "YPT" in wk.columns else ("team_plays", safe_sd),
    ypc_mean = ("YPC", safe_mean) if "YPC" in wk.columns else ("team_plays", safe_mean),
    ypc_sd   = ("YPC", safe_sd)   if "YPC" in wk.columns else ("team_plays", safe_sd),

    rec_tdrate  = ("rec_tdrate", safe_mean) if "rec_tdrate" in wk.columns else ("team_plays", safe_mean),
    rush_tdrate = ("rush_tdrate", safe_mean) if "rush_tdrate" in wk.columns else ("team_plays", safe_mean),

    plays_pg_mean = ("team_plays", safe_mean),
)

if "route_participation" in wk.columns:
    usage = usage.merge(gb.agg(route_participation=("route_participation", safe_mean)),
                        on=[c for c in gb_cols if c in usage.columns], how="left")
if "avg_sep" in wk.columns:
    usage = usage.merge(gb.agg(avg_sep=("avg_sep", safe_mean)),
                        on=[c for c in gb_cols if c in usage.columns], how="left")
if "yac_oe" in wk.columns:
    usage = usage.merge(gb.agg(yac_oe=("yac_oe", safe_mean)),
                        on=[c for c in gb_cols if c in usage.columns], how="left")

usage["target_share_k"] = 50.0
usage["rush_share_k"]   = 50.0

# TeamContext (per team)
tw = wk.dropna(subset=["team_plays"]).copy()
team_ctx = tw.groupby("team", as_index=False).agg(
    plays_pg_mean = ("team_plays","mean"),
    plays_pg_sd   = ("team_plays","std"),
    pass_rate_mean= ("team_pass_rate","mean")
)
team_ctx["pass_rate_k"] = 60.0
team_ctx["HFA_scalar"]  = 2.2

# DefenseAdjust (per defensive team)
if "opp" in wk.columns:
    by_def_pass = wk.groupby("opp", as_index=False).agg(
        rec_yards_sum = ("receiving_yards","sum") if "receiving_yards" in wk.columns else ("player_id","size"),
        targets_sum   = ("targets","sum") if "targets" in wk.columns else ("player_id","size")
    )
    by_def_pass["ypt_allowed"] = np.where(by_def_pass["targets_sum"]>0,
                                          by_def_pass["rec_yards_sum"]/by_def_pass["targets_sum"], np.nan)
    league_ypt = np.nanmean(by_def_pass["ypt_allowed"])
    by_def_pass["pass_def_mult"] = np.clip(by_def_pass["ypt_allowed"]/league_ypt, 0.85, 1.15)

    by_def_rush = wk.groupby("opp", as_index=False).agg(
        rush_yards_sum = ("rushing_yards","sum") if "rushing_yards" in wk.columns else ("player_id","size"),
        carries_sum    = ("rushing_attempts","sum") if "rushing_attempts" in wk.columns else ("player_id","size")
    )
    by_def_rush["ypc_allowed"] = np.where(by_def_rush["carries_sum"]>0,
                                          by_def_rush["rush_yards_sum"]/by_def_rush["carries_sum"], np.nan)
    league_ypc = np.nanmean(by_def_rush["ypc_allowed"])
    by_def_rush["rush_def_mult"] = np.clip(by_def_rush["ypc_allowed"]/league_ypc, 0.85, 1.15)

    defense_adj = by_def_pass[["opp","pass_def_mult"]].rename(columns={"opp":"team"})
    defense_adj = defense_adj.merge(by_def_rush[["opp","rush_def_mult"]].rename(columns={"opp":"team"}), on="team", how="outer")
else:
    defense_adj = pd.DataFrame({"team": team_ctx["team"], "pass_def_mult":1.0, "rush_def_mult":1.0})

# Save outputs
usage.to_parquet(OUT / "player_usage.parquet", index=False)
team_ctx.to_parquet(OUT / "team_context.parquet", index=False)
defense_adj.to_parquet(OUT / "defense_adjust.parquet", index=False)

print("✓ wrote", (OUT/"player_usage.parquet").resolve())
print("✓ wrote", (OUT/"team_context.parquet").resolve())
print("✓ wrote", (OUT/"defense_adjust.parquet").resolve())
