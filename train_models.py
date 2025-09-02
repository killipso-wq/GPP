# train_models.py — robust to missing 'pos'; trains LightGBM quantile models for WR/RB/TE
# Uses data produced by: bootstrap_2022_2024.py and build_features_2022_2024.py
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from joblib import dump

try:
    import lightgbm as lgb
except Exception as e:
    raise SystemExit(
        "LightGBM is required. Install inside venv:\n"
        "  python -m pip install lightgbm"
    )

DATA_DIR = Path("data_2022_2024")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ---------- helpers ----------
def pick(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def dk_points_non_qb(df: pd.DataFrame) -> pd.Series:
    """DraftKings scoring for RB/WR/TE using weekly columns if present."""
    rec = pd.to_numeric(df.get("receptions", 0), errors="coerce").fillna(0.0)
    rec_y = pd.to_numeric(df.get("receiving_yards", 0), errors="coerce").fillna(0.0)
    rec_td = pd.to_numeric(df.get("receiving_tds", 0), errors="coerce").fillna(0.0)

    ru_y = pd.to_numeric(df.get("rushing_yards", 0), errors="coerce").fillna(0.0)
    ru_td = pd.to_numeric(df.get("rushing_tds", 0), errors="coerce").fillna(0.0)

    # DK: 1 per reception, 0.1 per yard, 6 per TD
    pts = rec * 1.0 + 0.1 * rec_y + 6.0 * rec_td + 0.1 * ru_y + 6.0 * ru_td

    # 100+ yard bonuses (+3 each for rushing, receiving)
    pts += (rec_y >= 100).astype(float) * 3.0
    pts += (ru_y >= 100).astype(float) * 3.0

    # (We ignore QB passing stats on purpose; we train only WR/RB/TE.)
    return pts

def infer_pos_from_usage(usage: pd.DataFrame) -> pd.Series:
    """
    Crude but effective fallback:
      - RB if rush_share_mean >= 0.20 (heavy rush role)
      - TE if target_share_mean >= 0.08 and (avg_sep very low or YPT <= 8) -> proxy for TE profiles
      - else WR
    """
    ts = pd.to_numeric(usage.get("target_share_mean", np.nan), errors="coerce")
    rs = pd.to_numeric(usage.get("rush_share_mean", np.nan), errors="coerce")
    ypt = pd.to_numeric(usage.get("ypt_mean", np.nan), errors="coerce")
    sep = pd.to_numeric(usage.get("avg_sep", np.nan), errors="coerce")  # may be NaN

    pos = pd.Series(index=usage.index, dtype="object")
    pos[:] = "WR"

    rb_mask = (rs.fillna(0) >= 0.20)
    pos[rb_mask] = "RB"

    te_mask = (~rb_mask) & (ts.fillna(0) >= 0.08) & (
        (sep.notna() & (sep <= 2.0)) | (ypt.notna() & (ypt <= 8.0))
    )
    pos[te_mask] = "TE"

    return pos

def ensure_columns(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ---------- load inputs ----------
weekly_p = DATA_DIR / "weekly.parquet"
usage_p = DATA_DIR / "player_usage.parquet"
team_p  = DATA_DIR / "team_context.parquet"
def_p   = DATA_DIR / "defense_adjust.parquet"

for p in [weekly_p, usage_p, team_p, def_p]:
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run bootstrap_2022_2024.py and build_features_2022_2024.py first.")

weekly = pd.read_parquet(weekly_p)
usage  = pd.read_parquet(usage_p)
teamc  = pd.read_parquet(team_p)
defadj = pd.read_parquet(def_p)

# ---------- normalize weekly to needed columns ----------
name_col   = pick(weekly, ["player_name","player","full_name"], "player_name")
id_col     = pick(weekly, ["player_id","gsis_id","pfr_id"], "player_id")
team_col   = pick(weekly, ["team","recent_team","posteam"], "team")
opp_col    = pick(weekly, ["opponent_team","opponent","opp","defteam"], None)
season_col = pick(weekly, ["season","year"], "season")
week_col   = pick(weekly, ["week","game_week"], "week")

tgt_col    = pick(weekly, ["targets","target"], None)
rec_col    = pick(weekly, ["receptions","rec"], None)
recyd_col  = pick(weekly, ["receiving_yards","rec_yards","yards_receiving"], None)
rectd_col  = pick(weekly, ["receiving_tds","rec_tds"], None)

rushatt_col= pick(weekly, ["rushing_attempts","rush_attempts","carries"], None)
rushyd_col = pick(weekly, ["rushing_yards","rush_yards","yards_rushing"], None)
rushtd_col = pick(weekly, ["rushing_tds","rush_tds"], None)

pos_col    = pick(weekly, ["position","pos"], None)  # weekly position if present

need = [id_col, name_col, team_col, season_col, week_col]
for c in need:
    if c not in weekly.columns:
        raise ValueError(f"Weekly dataset missing required column: {c}")

keep = [id_col,name_col,team_col,season_col,week_col] + \
       [c for c in [opp_col, pos_col, tgt_col, rec_col, recyd_col, rectd_col, rushatt_col, rushyd_col, rushtd_col] if c]

wk = weekly[keep].copy()
wk.rename(columns={
    id_col:"player_id", name_col:"player_name", team_col:"team",
    season_col:"season", week_col:"week"
}, inplace=True)
if opp_col:      wk.rename(columns={opp_col:"opp"}, inplace=True)
if pos_col:      wk.rename(columns={pos_col:"pos"}, inplace=True)
if tgt_col:      wk.rename(columns={tgt_col:"targets"}, inplace=True)
if rec_col:      wk.rename(columns={rec_col:"receptions"}, inplace=True)
if recyd_col:    wk.rename(columns={recyd_col:"receiving_yards"}, inplace=True)
if rectd_col:    wk.rename(columns={rectd_col:"receiving_tds"}, inplace=True)
if rushatt_col:  wk.rename(columns={rushatt_col:"rushing_attempts"}, inplace=True)
if rushyd_col:   wk.rename(columns={rushyd_col:"rushing_yards"}, inplace=True)
if rushtd_col:   wk.rename(columns={rushtd_col:"rushing_tds"}, inplace=True)

# Clean numeric
for c in ["targets","receptions","receiving_yards","receiving_tds",
          "rushing_attempts","rushing_yards","rushing_tds"]:
    if c in wk.columns:
        wk[c] = pd.to_numeric(wk[c], errors="coerce").fillna(0)

wk["team"] = wk["team"].astype(str).str.upper().str.strip()
if "opp" in wk.columns:
    wk["opp"] = wk["opp"].astype(str).str.upper().str.strip()

# DraftKings target for non-QBs
wk["fpts_dk"] = dk_points_non_qb(wk)

# ---------- attach usage/team/defense, infer pos if missing ----------
usage = usage.copy()
ensure_columns(usage, ["player_id","player_name","team","target_share_mean","rush_share_mean",
                       "ypt_mean","ypc_mean","rec_tdrate","rush_tdrate",
                       "route_participation","avg_sep","yac_oe","pos"])

if "pos" not in usage.columns or usage["pos"].isna().all():
    usage["pos"] = infer_pos_from_usage(usage)

# keep only WR/RB/TE in usage (so we don't mix QBs/kickers/def)
usage = usage[usage["pos"].isin(["WR","RB","TE"])].copy()

# Merge usage onto weekly by player_id (primary) and fallback by name+team
wk = wk.merge(usage.drop_duplicates(subset=["player_id"])[
    ["player_id","pos","target_share_mean","rush_share_mean",
     "ypt_mean","ypc_mean","rec_tdrate","rush_tdrate",
     "route_participation","avg_sep","yac_oe"]
], on="player_id", how="left")

# Add team context (pass rate proxy)
wk = wk.merge(teamc[["team","pass_rate_mean","plays_pg_mean"]], on="team", how="left")

# Add opponent defense multipliers (optional)
if "opp" in wk.columns:
    wk = wk.merge(defadj.rename(columns={"team":"opp"})[["opp","pass_def_mult","rush_def_mult"]],
                  on="opp", how="left")
else:
    wk["pass_def_mult"] = np.nan
    wk["rush_def_mult"] = np.nan

# Finalize pos (if still missing, infer from weekly behavior)
if "pos" not in wk.columns or wk["pos"].isna().all():
    # infer from per-week behavior: heavy rush -> RB; otherwise WR/TE by ypt
    rs = pd.to_numeric(wk.get("rushing_attempts", 0), errors="coerce").fillna(0)
    ts = pd.to_numeric(wk.get("targets", 0), errors="coerce").fillna(0)
    ypt = pd.to_numeric(wk.get("ypt_mean", np.nan), errors="coerce")

    pos = pd.Series("WR", index=wk.index, dtype="object")
    pos[(rs >= 5) & (ts < 3)] = "RB"
    pos[(ts >= 3) & (ypt.notna()) & (ypt <= 8.0)] = "TE"
    wk["pos"] = pos

# Filter to WR/RB/TE only
wk = wk[wk["pos"].isin(["WR","RB","TE"])].copy()

# Remove obvious empties (no stats at all)
wk = wk[~((wk["receptions"]==0) & (wk["rushing_yards"]==0) & (wk["receiving_yards"]==0))].copy()

# ---------- build feature matrix ----------
FEATURES = [
    # usage-based
    "target_share_mean","rush_share_mean","ypt_mean","ypc_mean",
    "rec_tdrate","rush_tdrate","route_participation","avg_sep","yac_oe",
    # team/opp context
    "pass_rate_mean","plays_pg_mean","pass_def_mult","rush_def_mult",
    # raw weekly signals (light leakage ok for training label formation)
    "targets","rushing_attempts"
]
wk = ensure_columns(wk, FEATURES)
X_all = wk[FEATURES].copy()
X_all = X_all.fillna(0.0)
y_all = wk["fpts_dk"].astype(float)

# ---------- train quantile models per position ----------
RANDOM_STATE = 1337
POS_LIST = ["WR","RB","TE"]
ALPHAS = [0.10, 0.50, 0.90]

def train_pos(df_pos: pd.DataFrame, pos: str):
    X = df_pos[FEATURES].values
    y = df_pos["fpts_dk"].values

    if len(df_pos) < 200:
        print(f"[{pos}] Not enough rows ({len(df_pos)}) — skipping.")
        return

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # train each quantile
    models = {}
    for a in ALPHAS:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=a,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        mae = mean_absolute_error(y_te, pred)
        print(f"[{pos}] alpha={a:.2f}  MAE={mae:.3f}  (n={len(y_te)})")
        models[a] = model
        dump(model, MODELS_DIR / f"lgbm_{pos}_q{int(a*100)}.pkl")

    # save feature list for inference
    meta = {
        "features": FEATURES,
        "alphas": ALPHAS,
        "pos": pos,
    }
    dump(meta, MODELS_DIR / f"lgbm_{pos}_meta.pkl")
    print(f"[{pos}] Saved models to {MODELS_DIR.resolve()}")

for pos in POS_LIST:
    dfp = wk[wk["pos"] == pos].copy()
    if dfp.empty:
        print(f"[{pos}] no rows, skipping.")
        continue
    train_pos(dfp, pos)

print("\nDone. Models are in:", MODELS_DIR.resolve())
