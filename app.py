# app.py — DFS NFL Monte Carlo — Weekly Simulator + Optimizer
# VERSION 17: Fix NameError for ALL_FAST_SURFACE + robust env logic + safer merges.

from __future__ import annotations
import math
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---- Fast/Dome (or similar “fast surface”) teams (Option A: inline constant) ----
# Note: keep TEAM codes UPPERCASE to match your CSVs
ALL_FAST_SURFACE: set[str] = {
    "ATL", "ARI", "DAL", "DET", "HOU", "IND", "LAC", "LAR", "LV", "MIN", "NO"
}

# ---- Try to import your existing simulation engine and optimizer (best-effort) ----
try:
    import simulation_engine as simeng
except Exception as e:
    simeng = None
    st.sidebar.warning(f"simulation_engine import failed: {e}")

try:
    import optimizer_gpp as optgpp
except Exception as e:
    optgpp = None
    st.sidebar.info(f"optimizer_gpp import failed (you can still simulate): {e}")

# =========================================================
# Utilities
# =========================================================

def _nx(row: pd.Series, *names: str):
    """Return the first non-null value among possible column aliases in a row."""
    for n in names:
        if n in row and not pd.isna(row[n]):
            return row[n]
    return None

def _choose_series(df: pd.DataFrame, *candidates: str, default=None, coerce_numeric=False) -> pd.Series:
    """Pick the first existing column among candidates; else return default-valued Series."""
    for c in candidates:
        if c in df.columns:
            s = df[c]
            if coerce_numeric:
                s = pd.to_numeric(s, errors="coerce")
            return s
    if isinstance(default, (int, float)) or default is None:
        out = pd.Series(default, index=df.index, dtype="float64" if isinstance(default, (int, float)) else "object")
    else:
        out = pd.Series(default, index=df.index)
    return out

def _derive_is_home(df: pd.DataFrame) -> pd.Series:
    """
    Try to derive home/away:
      - If column HOME exists (bool / 0-1), use it.
      - Else infer from OPP values like '@GB' (away) or 'GB' (home).
    """
    if "HOME" in df.columns:
        return df["HOME"].astype(bool)

    # Fallback: OPP like '@GB' = away, else = home
    opp = df["OPP"].astype(str) if "OPP" in df.columns else pd.Series("", index=df.index)
    return ~opp.str.strip().str.startswith("@")

def _safe_team_upper(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper()

# =========================================================
# Simulation wrapper
# =========================================================

def run_simulation_with_best_effort(raw_df: pd.DataFrame, sims_per_matchup: int = 10_000) -> pd.DataFrame:
    """
    Calls your simulation_engine if present; otherwise produces a trivial table
    so the UI remains usable.
    Expected output schema (flexible): at least columns for 'PLAYER' and some sim stats.
    """
    try:
        if simeng is not None and hasattr(simeng, "simulate_players"):
            # If your engine exposes simulate_players(DataFrame, sims) returning per-player stats
            sim = simeng.simulate_players(raw_df, sims=sims_per_matchup)
            if not isinstance(sim, pd.DataFrame):
                raise RuntimeError("simulate_players did not return a DataFrame.")
            return sim

        # Alternate known entrypoints (if your engine has different names)
        if simeng is not None and hasattr(simeng, "simulate"):
            sim = simeng.simulate(raw_df, sims=sims_per_matchup)
            if not isinstance(sim, pd.DataFrame):
                raise RuntimeError("simulate did not return a DataFrame.")
            return sim

        # Fallback: make a minimal table so downstream steps still work
        tmp = pd.DataFrame()
        tmp["PLAYER"] = _choose_series(raw_df, "PLAYER", "Name", "player", default="Unknown").astype(str)
        # Create dummy sim stats based on any projection we can find
        base = _choose_series(raw_df, "PROJ_BASE", "PROJ", "FPTS", "DK_PROJ", "Proj", default=0.0, coerce_numeric=True).fillna(0.0)
        tmp["SIM_MEAN"] = base
        tmp["SIM_P50"]  = base
        tmp["SIM_P10"]  = (base * 0.6).round(2)
        tmp["SIM_P90"]  = (base * 1.4).round(2)
        tmp["Boom%"]    = 0.0
        return tmp

    except Exception as e:
        st.error(f"Advanced simulation failed: {e}")
        return pd.DataFrame()

# =========================================================
# Final table assembly (adds environment/home tweaks and merges sim → displayable table)
# =========================================================

def compute_final_gpp_table(raw_df: pd.DataFrame, sim_table: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # Normalize key ID columns
    if "PLAYER" not in df.columns:
        # Create PLAYER from first available name-like column
        df["PLAYER"] = _choose_series(df, "Name", "player", "PLAYER", default="Unknown").astype(str)

    df["PLAYER"] = df["PLAYER"].astype(str)

    if "TEAM" not in df.columns:
        df["TEAM"] = _choose_series(df, "Tm", "TEAM", "team", default="").astype(str)

    df["TEAM"] = _safe_team_upper(df["TEAM"])

    # Merge sim stats (by PLAYER; if you have a unique ID, swap it here)
    sim = sim_table.copy()
    if "PLAYER" not in sim.columns:
        # Attempt to find a compatible key
        key = next((c for c in ["Name", "player", "Player"] if c in sim.columns), None)
        if key is None:
            raise RuntimeError("Simulation table lacks a PLAYER-like column to merge on.")
        sim = sim.rename(columns={key: "PLAYER"})

    sim_cols = [c for c in sim.columns if c != "PLAYER"]
    merged = df.merge(sim[["PLAYER"] + sim_cols], on="PLAYER", how="left")

    # Derive base projection (uses user/CSV projections if present; else fallback to SIM_MEAN)
    proj_base = _choose_series(
        merged,
        "PROJ_BASE", "PROJ", "FPTS", "DK_PROJ", "Proj", "Projection",
        default=np.nan, coerce_numeric=True
    )
    sim_mean = _choose_series(merged, "SIM_MEAN", "mean", "SIM_MEAN_FPTS", default=np.nan, coerce_numeric=True)

    merged["PROJ_BASE"] = proj_base.fillna(sim_mean).fillna(0.0)

    # Home/away + fast-surface (dome) bonus — robust and non-crashing
    is_home = _derive_is_home(merged)

    fast = set(ALL_FAST_SURFACE) if "ALL_FAST_SURFACE" in globals() and isinstance(ALL_FAST_SURFACE, (set, list, tuple)) else set()
    is_dome_team = _safe_team_upper(merged["TEAM"]).isin(fast)

    # Tunable knobs: small, conservative bumps
    dome_bonus = 0.03  # +3% in dome/fast surface
    home_bonus = 0.01  # +1% at home

    bonus_pct = np.where(is_dome_team, dome_bonus, 0.00) + np.where(is_home, home_bonus, 0.00)
    merged["PROJ_ENV"] = merged["PROJ_BASE"] * (1.0 + bonus_pct)

    # Create clean, Streamlit-friendly columns for display
    # Grab common sim columns if available
    merged["SIM_P50"] = _choose_series(merged, "SIM_P50", "p50", "median", default=np.nan, coerce_numeric=True)
    merged["SIM_P10"] = _choose_series(merged, "SIM_P10", "p10", default=np.nan, coerce_numeric=True)
    merged["SIM_P90"] = _choose_series(merged, "SIM_P90", "p90", default=np.nan, coerce_numeric=True)
    merged["Boom%"]   = _choose_series(merged, "Boom%", "BOOM%", "boom", default=np.nan, coerce_numeric=True)

    # Salary + position (DraftKings-style) if present
    merged["POS"] = _choose_series(merged, "POS", "Pos", default="")
    merged["SAL"] = _choose_series(merged, "SAL", "Salary", "DK_SAL", default=np.nan, coerce_numeric=True)

    # Order for readability
    display_cols = [c for c in [
        "PLAYER", "POS", "TEAM", "SAL",
        "PROJ_BASE", "PROJ_ENV",
        "SIM_MEAN", "SIM_P10", "SIM_P50", "SIM_P90", "Boom%",
        "OPP"
    ] if c in merged.columns]

    # Fill remaining with anything else useful
    for c in ["OU", "SPRD", "WIN%", "RST%"]:
        if c in merged.columns and c not in display_cols:
            display_cols.append(c)

    return merged[display_cols].sort_values(by=["POS", "PROJ_ENV"], ascending=[True, False]).reset_index(drop=True)

# =========================================================
# Optimizer driver (optional)
# =========================================================

def run_optimizer_if_available(final_tbl: pd.DataFrame, num_lineups: int = 20) -> pd.DataFrame | None:
    """
    If your optimizer_gpp.py exposes a public API, call it here.
    Otherwise, return None gracefully.
    """
    try:
        if optgpp is None:
            return None

        # Example API discovery (adjust to your actual function)
        if hasattr(optgpp, "generate_lineups"):
            return optgpp.generate_lineups(final_tbl, num_lineups=num_lineups)

        if hasattr(optgpp, "optimize"):
            return optgpp.optimize(final_tbl, k=num_lineups)

        return None
    except Exception as e:
        st.error(f"Optimizer error: {e}")
        return None

# =========================================================
# Streamlit App
# =========================================================

st.set_page_config(page_title="DFS NFL — Monte Carlo Simulator & GPP Optimizer", layout="wide")
st.title("🏈 DFS NFL — Monte Carlo Simulator & Optimizer")

st.markdown(
    "Upload your weekly **players.csv** (RAW slate/player pool). "
    "Then run the simulation and (optionally) generate lineups."
)

with st.sidebar:
    st.subheader("Simulation Settings")
    sims = st.number_input("Sims per matchup", min_value=1000, max_value=100_000, step=1000, value=10_000)
    st.caption("Higher = smoother distributions but slower.")

uploaded = st.file_uploader("Upload players.csv", type=["csv"])

if "final_gpp_table" not in st.session_state:
    st.session_state.final_gpp_table = None

if uploaded is not None:
    st.header("Step 2: Run Simulation")
    if st.button("▶️ Run Monte Carlo Simulation", type="primary", use_container_width=True):
        with st.spinner("Running advanced simulation..."):
            raw_df = pd.read_csv(uploaded)
            sim_table = run_simulation_with_best_effort(raw_df, sims_per_matchup=int(sims))
            if sim_table is None or sim_table.empty:
                st.warning("Simulation returned no rows (check your CSV columns).")
            final_tbl = compute_final_gpp_table(raw_df, sim_table)
            st.session_state.final_gpp_table = final_tbl
            st.success("✅ Simulation complete! Scroll down to analyze results and (optionally) optimize lineups.")

if st.session_state.get("final_gpp_table") is not None:
    st.header("Step 3: Analyze Results")
    final_tbl = st.session_state.final_gpp_table

    st.dataframe(
        final_tbl.style.format({
            "SAL": "{:,.0f}",
            "PROJ_BASE": "{:.2f}",
            "PROJ_ENV": "{:.2f}",
            "SIM_MEAN": "{:.2f}",
            "SIM_P10": "{:.2f}",
            "SIM_P50": "{:.2f}",
            "SIM_P90": "{:.2f}",
            "Boom%": "{:.1f}",
        }),
        use_container_width=True,
        height=520
    )

    st.header("Step 4 (Optional): Generate Lineups")
    cols = st.columns(3)
    with cols[0]:
        k = st.number_input("Number of lineups", min_value=1, max_value=150, step=1, value=20)
    with cols[1]:
        enforce_salary_cap = st.checkbox("Enforce DK salary cap (50,000)", value=True)
    with cols[2]:
        show_only_top = st.number_input("Show top N", min_value=1, max_value=150, step=1, value=20)

    if st.button("🧮 Optimize Lineups", type="secondary"):
        with st.spinner("Optimizing…"):
            lineups = run_optimizer_if_available(final_tbl, int(k))
            if lineups is None or (isinstance(lineups, pd.DataFrame) and lineups.empty):
                st.warning("No optimizer available or it returned no lineups. Check optimizer_gpp.py.")
            else:
                # If optimizer doesn't enforce cap, you can filter here
                if enforce_salary_cap and "Salary" in lineups.columns:
                    lineups = lineups[lineups["Salary"] <= 50_000]

                st.subheader("Lineups")
                st.dataframe(lineups.head(int(show_only_top)), use_container_width=True)

    st.header("Step 5: Export")
    exp_cols = st.columns(2)
    with exp_cols[0]:
        if st.button("⬇️ Download Final Table (CSV)"):
            csv_bytes = final_tbl.to_csv(index=False).encode("utf-8")
            st.download_button("Save final_gpp_table.csv", data=csv_bytes, file_name="final_gpp_table.csv", mime="text/csv")
    with exp_cols[1]:
        st.caption("You can also export lineups above (if optimizer returns a DataFrame).")
