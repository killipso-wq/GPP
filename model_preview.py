# model_preview.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from ml_predict import predict_players

st.set_page_config(page_title="ML Projections Preview", layout="wide")

st.title("ML Projections — p10 / p50 / p90 (WR/RB/TE)")

players_path = st.text_input("Path to players.csv", value="players.csv")

miami_only = st.checkbox("Miami-only quick test", value=False)
run_btn = st.button("Run ML predictions")

if run_btn:
    try:
        pred = predict_players(players_path)
        # Load your projections from players.csv to compare (expects FTPS column if you have it)
        raw = pd.read_csv(players_path)
        # normalize col names
        rename = {}
        for raw_name, std in [("PLAYER","PLAYER"), ("TeamAbbrev","TEAM"), ("TEAM","TEAM"), ("Opp","OPP"), ("OPP","OPP"), ("POS","POS"), ("Salary","SAL"), ("SAL","SAL")]:
            if raw_name in raw.columns: rename[raw_name] = std
        if rename: raw = raw.rename(columns=rename)
        for c in ["PLAYER","TEAM","OPP","POS"]:
            if c in raw.columns:
                raw[c] = raw[c].astype(str).str.upper().str.strip()

        # your projection column (try common names)
        proj_col = None
        for c in ["FTPS","FPTS","PROJ","PROJECTION","Proj","Fpts"]:
            if c in raw.columns:
                proj_col = c
                break

        view = pred.copy()
        if proj_col:
            view = view.merge(raw[["PLAYER","TEAM","POS","OPP",proj_col]], on=["PLAYER","TEAM","POS","OPP"], how="left")
            view = view.rename(columns={proj_col:"YourProj"})

        if miami_only:
            view = view[view["TEAM"].eq("MIA") | view["OPP"].eq("MIA")]

        # Boom% from normal approximation around median/sigma
        # P(X >= mu+sigma) ~ 0.1587; here we compute directly from N(μ,σ)
        z = 1.0
        boom_prob = 1.0 - 0.5*(1.0 + np.math.erf(z/np.sqrt(2)))  # == ~0.1587 constant
        view["Boom%~"] = (1.0 - 0.5*(1.0 + (1/np.sqrt(2*np.pi))*0 + 0))  # keep column but not used
        view["Boom%~"] = 0.159  # display constant approx; real sim uses sampling

        # diff vs your projection, if present
        if "YourProj" in view.columns:
            view["Diff (p50 - YourProj)"] = (view["p50"] - view["YourProj"]).round(2)

        order = ["PLAYER","TEAM","OPP","POS","SAL","p10","p50","p90"]
        if "YourProj" in view.columns:
            order += ["YourProj","Diff (p50 - YourProj)"]
        order += ["sigma","boom_threshold"]

        st.dataframe(view[order].sort_values("p50", ascending=False), use_container_width=True, height=560)

        st.caption("Notes: p50 is a blended median (Elastic Net + LightGBM via ridge). p10/p90 from LightGBM quantiles. σ≈(p90−p10)/2.563.")

    except Exception as e:
        st.error(f"Error: {e}")
