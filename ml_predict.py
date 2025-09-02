# ml_predict.py
# Load saved models (models/WR|RB|TE) and predict p10/p50/p90 for players in players.csv
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

DATA = Path("data_2022_2024")
MODELS = Path("models")

def _read_players_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize common columns
    rename = {}
    for raw, std in [("PLAYER","PLAYER"), ("TeamAbbrev","TEAM"), ("TEAM","TEAM"), ("Opp","OPP"), ("OPP","OPP"), ("POS","POS"), ("Salary","SAL"), ("SAL","SAL")]:
        if raw in df.columns: rename[raw] = std
    if rename: df = df.rename(columns=rename)
    for c in ["PLAYER","TEAM","OPP","POS"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    if "SAL" in df.columns:
        df["SAL"] = pd.to_numeric(df["SAL"], errors="coerce")
    return df

def _safe_load_pos_models(pos: str):
    d = MODELS / pos
    if not d.exists():
        return None
    try:
        lgb_p10 = joblib.load(d / "lgb_p10.joblib")
        lgb_p50 = joblib.load(d / "lgb_p50.joblib")
        lgb_p90 = joblib.load(d / "lgb_p90.joblib")
        enet    = joblib.load(d / "enet_p50.joblib")
        blender = joblib.load(d / "blender_ridge.joblib")
        meta    = json.load(open(d / "meta.json","r",encoding="utf-8"))
        return dict(lgb_p10=lgb_p10, lgb_p50=lgb_p50, lgb_p90=lgb_p90, enet=enet, blender=blender, meta=meta)
    except Exception:
        return None

def _load_feature_tables():
    usage  = pd.read_parquet(DATA/"player_usage.parquet")
    tctx   = pd.read_parquet(DATA/"team_context.parquet")
    dadj   = pd.read_parquet(DATA/"defense_adjust.parquet")
    # minimal subset
    return usage, tctx, dadj

def _build_features(players_df: pd.DataFrame, usage: pd.DataFrame, tctx: pd.DataFrame, dadj: pd.DataFrame, feat_list):
    # try to merge by player_name + TEAM
    use = usage.copy()
    if "player_name" not in use.columns:
        # fallback: use whatever name column exists
        for c in ["PLAYER","player","name"]:
            if c in use.columns:
                use = use.rename(columns={c:"player_name"})
                break
    cols_we_want = ["player_name","team","pos"] + [c for c in feat_list if c in use.columns]
    u = use[cols_we_want].drop_duplicates(subset=["player_name","team"], keep="last").copy()
    u["team"] = u["team"].astype(str).str.upper().str.strip()
    if "pos" in u.columns:
        u["pos"] = u["pos"].astype(str).str.upper().str.strip()

    df = players_df.copy()
    df = df.rename(columns={"TEAM":"team","OPP":"opp","POS":"pos"})
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["opp"]  = df["opp"].astype(str).str.upper().str.strip()
    df["pos"]  = df["pos"].astype(str).str.upper().str.strip()

    # Merge usage by name+team
    merged = df.merge(u, left_on=["PLAYER","team"], right_on=["player_name","team"], how="left", suffixes=("","_u"))

    # Team context
    merged = merged.merge(tctx[["team","plays_pg_mean","pass_rate_mean"]], on="team", how="left")
    # Defense (by opponent)
    merged = merged.merge(dadj.rename(columns={"team":"opp"}), on="opp", how="left")

    # keep only needed features
    all_feats = list(dict.fromkeys(feat_list + ["plays_pg_mean","pass_rate_mean","pass_def_mult","rush_def_mult"]))
    for c in all_feats:
        if c not in merged.columns:
            merged[c] = np.nan

    return merged, all_feats

def predict_players(players_csv="players.csv") -> pd.DataFrame:
    players_df = _read_players_csv(Path(players_csv))
    usage, tctx, dadj = _load_feature_tables()

    out_rows = []
    for pos in ["WR","RB","TE"]:
        models = _safe_load_pos_models(pos)
        if models is None:
            continue
        feat_list = models["meta"]["features"]
        merged, all_feats = _build_features(players_df[players_df["POS"].str.upper()==pos], usage, tctx, dadj, feat_list)

        # fill with medians from training
        meds = models["meta"]["medians"]
        for c in all_feats:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
            fillv = meds.get(c, np.nan)
            merged[c] = merged[c].fillna(fillv)

        X = merged[all_feats].values
        p10 = models["lgb_p10"].predict(X)
        p50_lgb = models["lgb_p50"].predict(X)
        p90 = models["lgb_p90"].predict(X)
        p50_en = models["enet"].predict(X)

        Z = np.column_stack([p50_en, p50_lgb])
        p50_blend = models["blender"].predict(Z)

        sub = merged[["PLAYER","team","opp","pos","SAL"]].copy()
        sub.rename(columns={"team":"TEAM","opp":"OPP","pos":"POS"}, inplace=True)
        sub["p10"] = p10
        sub["p50"] = p50_blend
        sub["p90"] = p90

        # boom% (>= mu + sigma) using normal approx: sigma ~ (p90-p10)/2.563
        sigma = (p90 - p10) / 2.563
        boom_thr = sub["p50"] + sigma
        # for a normal, P(X>=mu+sigma)=~0.1587; but we compute from draws when needed
        sub["boom_threshold"] = boom_thr
        sub["sigma"] = sigma.clip(lower=0.1)  # floor to avoid zero
        out_rows.append(sub)

    if not out_rows:
        raise RuntimeError("No position models found in models/. Did you run train_models.py?")

    pred = pd.concat(out_rows, ignore_index=True)
    # order & friendly names
    pred = pred[["PLAYER","TEAM","OPP","POS","SAL","p10","p50","p90","sigma","boom_threshold"]]
    return pred
