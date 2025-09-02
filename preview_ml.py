# preview_ml.py — ML p10/p50/p90 preview (handles FLEX, skips unsupported POS)
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

DATA = Path("data_2022_2024")
MODELS = Path("models")

ALL_FEATURES = [
    "target_share_mean","rush_share_mean","ypt_mean","ypc_mean",
    "rec_tdrate","rush_tdrate","route_participation","avg_sep","yac_oe",
    "pass_rate_mean","plays_pg_mean","pass_def_mult","rush_def_mult",
    "targets","rushing_attempts",
]
OPTIONAL_FEATURES = {"route_participation","avg_sep","yac_oe"}

def pick(df, names, default=None):
    for n in names:
        if n in df.columns:
            return n
    return default

def normalize_pos(p):
    if not isinstance(p, str):
        return None
    u = p.upper().strip()
    if "RB" in u: return "RB"
    if "WR" in u: return "WR"
    if "TE" in u: return "TE"
    if "QB" in u: return "QB"   # not modeled yet
    if "DST" in u or "DEF" in u: return "DST"
    return None

def load_pos_models(pos):
    meta = load(MODELS / f"lgbm_{pos}_meta.pkl")
    m10  = load(MODELS / f"lgbm_{pos}_q10.pkl")
    m50  = load(MODELS / f"lgbm_{pos}_q50.pkl")
    m90  = load(MODELS / f"lgbm_{pos}_q90.pkl")
    return meta, m10, m50, m90

# load available pos bundles
SUPPORTED = ["WR","RB","TE"]
MODELB = {p: load_pos_models(p) for p in SUPPORTED if (MODELS / f"lgbm_{p}_meta.pkl").exists()}

# feature tables
usage  = pd.read_parquet(DATA / "player_usage.parquet")
teamc  = pd.read_parquet(DATA / "team_context.parquet")
defadj = pd.read_parquet(DATA / "defense_adjust.parquet")

usage["player_name_key"] = usage["player_name"].astype(str).str.upper().str.strip()
usage["team"] = usage["team"].astype(str).str.upper().str.strip()
if "pos" not in usage.columns or usage["pos"].isna().all():
    usage["pos"] = "WR"

present_optional = [c for c in OPTIONAL_FEATURES if c in usage.columns]
team_pos_cols = [
    c for c in [
        "target_share_mean","rush_share_mean","ypt_mean","ypc_mean",
        "rec_tdrate","rush_tdrate"
    ] + present_optional
    if c in usage.columns
]
team_pos_avg = usage.groupby(["team","pos"], as_index=False)[team_pos_cols].mean()

# read slate
players_csv = Path("players.csv")
if not players_csv.exists():
    alts = list(Path(".").glob("*players*.csv"))
    if alts:
        players_csv = alts[0]
players = pd.read_csv(players_csv)

player_col = pick(players, ["PLAYER","Name","player","player_name"], "PLAYER")
team_col   = pick(players, ["TEAM","Team","team"], "TEAM")
opp_col    = pick(players, ["OPP","Opp","opponent","DEF","opp"], None)
pos_col    = pick(players, ["POS","Position","pos"], None)
sal_col    = pick(players, ["SAL","Salary","salary"], None)
fpts_col   = pick(players, ["FPTS","Proj","Projection","fpts"], None)

players = players.rename(columns={
    player_col: "PLAYER", team_col: "TEAM",
    **({opp_col: "OPP"} if opp_col else {}),
    **({pos_col: "POS"} if pos_col else {}),
    **({sal_col: "SAL"} if sal_col else {}),
    **({fpts_col: "FPTS"} if fpts_col else {}),
})
players["PLAYER_KEY"] = players["PLAYER"].astype(str).str.upper().str.strip()
players["TEAM"] = players["TEAM"].astype(str).str.upper().str.strip()
if "OPP" in players.columns:
    players["OPP"] = players["OPP"].astype(str).str.upper().str.strip()
players["POS_NORM"] = players["POS"].apply(normalize_pos) if "POS" in players.columns else None

# join usage
merged = players.merge(
    usage[["player_id","player_name","player_name_key","team","pos"] + team_pos_cols],
    left_on=["PLAYER_KEY","TEAM"], right_on=["player_name_key","team"], how="left"
)

# choose POS for modeling
if "POS_NORM" in players.columns:
    merged["POS_NORM"] = players["POS_NORM"]
else:
    merged["POS_NORM"] = merged["pos"].apply(normalize_pos)

# backfill player features with team/pos averages
def fill_from_team_pos(row):
    t = row["TEAM"]
    p = (row.get("POS_NORM") or row.get("pos") or "WR")
    tpl = team_pos_avg[(team_pos_avg["team"]==t) & (team_pos_avg["pos"]==p)]
    if not tpl.empty:
        for c in team_pos_cols:
            if pd.isna(row.get(c)):
                row[c] = tpl.iloc[0][c]
    return row

merged = merged.apply(fill_from_team_pos, axis=1)

# attach team context and opponent defense
merged = merged.merge(
    teamc[["team","pass_rate_mean","plays_pg_mean"]].rename(columns={"team":"TEAM"}),
    on="TEAM", how="left"
)
if "OPP" in merged.columns:
    merged = merged.merge(defadj.rename(columns={"team":"OPP"}), on="OPP", how="left")
else:
    merged["pass_def_mult"] = np.nan
    merged["rush_def_mult"] = np.nan

# zeros for count features
merged["targets"] = 0.0
merged["rushing_attempts"] = 0.0

def row_to_features(row):
    feat = {f: 0.0 for f in ALL_FEATURES}
    for f in ALL_FEATURES:
        if f in row.index and pd.notna(row[f]):
            feat[f] = float(row[f])
    return pd.DataFrame([feat])[ALL_FEATURES]

def predict_row(row):
    pos = str(row.get("POS_NORM") or "").upper()
    if pos not in MODELB:
        return pd.Series({"p10": np.nan, "p50": np.nan, "p90": np.nan})
    meta, m10, m50, m90 = MODELB[pos]
    X = row_to_features(row)
    return pd.Series({
        "p10": float(m10.predict(X)[0]),
        "p50": float(m50.predict(X)[0]),
        "p90": float(m90.predict(X)[0]),
    })

preds = merged.apply(predict_row, axis=1)
out = pd.concat([merged, preds], axis=1)

# implied sd only when we have both tails
Z10, Z90 = -1.2815515655446004, 1.2815515655446004
out["sd_implied"] = np.where(out[["p10","p90"]].notna().all(axis=1),
                             (out["p90"] - out["p10"]) / (Z90 - Z10),
                             np.nan)

# display/save
cols = ["PLAYER","TEAM","OPP","POS","POS_NORM","SAL","FPTS","p10","p50","p90","sd_implied"]
cols = [c for c in cols if c in out.columns]
out_disp = out[cols].sort_values(["POS_NORM","p50"], ascending=[True, False])

Path("out_csv").mkdir(exist_ok=True)
out_path = Path("out_csv/ml_predictions_preview.csv")
out_disp.to_csv(out_path, index=False)
print(f"✓ Wrote {out_path}  ({len(out_disp)} rows)")
print(f"Predicted rows (supported POS {list(MODELB.keys())}): {(out_disp['p50'].notna()).sum()}")

try:
    import streamlit as st
    st.title("ML Projections Preview (p10 / p50 / p90)")
    st.caption("LightGBM quantiles for WR/RB/TE; FLEX mapped automatically. QBs not modeled yet.")
    st.dataframe(out_disp, use_container_width=True, height=640)
except Exception:
    pass
