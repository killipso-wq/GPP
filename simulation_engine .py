# simulation_engine.py — Monte Carlo engine (enhanced)
# Consumes ML quantiles and/or user projections (+ optional props), produces per-player FPTS distributions
# with correlation, game-script effects, and TD-specific modeling.

from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULT_SIMS = 10_000

# -------------------------
# Distribution utilities
# -------------------------

def _norm_params_from_quantiles(p10: float, p50: float, p90: float) -> tuple[float, float]:
    """For a roughly normal distribution, (p90 - p10) ≈ 2 * 1.28155 * sd = 2.5631 * sd."""
    sd = max((float(p90) - float(p10)) / 2.5631031310892008, 1e-6)
    return float(p50), sd

def _splitnorm_params_from_quantiles(p10: float, p50: float, p90: float) -> tuple[float, float, float]:
    """Derive split-normal sd's from quantiles using z(0.9)=1.28155."""
    z10 = 1.2815515655446004
    sd_left = max((float(p50) - float(p10)) / z10, 1e-6)
    sd_right = max((float(p90) - float(p50)) / z10, 1e-6)
    return float(p50), sd_left, sd_right

def _sample_player_points(mu: float, sd: float, n: int) -> np.ndarray:
    draws = np.random.normal(loc=float(mu), scale=float(sd), size=int(n))
    return np.clip(draws, 0.0, 80.0)

def _sample_split_normal(mu: float, sd_left: float, sd_right: float, n: int) -> np.ndarray:
    z = np.random.normal(loc=0.0, scale=1.0, size=int(n))
    out = float(mu) + np.where(z < 0, z * float(sd_left), z * float(sd_right))
    return np.clip(out, 0.0, 80.0)

# -------------------------
# Environment & props helpers
# -------------------------

def _env_multiplier_from_implied(imp_pts: float | None, pos: str | None) -> float:
    """Scale by team implied points vs league avg (22). Clamp to [0.85, 1.20]."""
    if imp_pts is None or pd.isna(imp_pts):
        return 1.0
    league = 22.0
    base = float(np.clip(float(imp_pts) / league, 0.85, 1.20))
    if pos in ("WR", "TE"):
        return base
    if pos == "RB":
        # RBs less sensitive than pass catchers
        return 0.5 * (1.0 + base)
    return 1.0

def _defense_multiplier(row: pd.Series) -> float:
    """If defense multipliers provided via ML join, nudge projections a bit."""
    m = 1.0
    posv = (str(row.get("POS") if "POS" in row else row.get("pos") or "")).upper()
    if posv in ("WR", "TE") and not pd.isna(row.get("pass_def_mult")):
        m *= float(np.clip(row["pass_def_mult"], 0.90, 1.10))
    if posv == "RB" and not pd.isna(row.get("rush_def_mult")):
        m *= float(np.clip(row["rush_def_mult"], 0.90, 1.10))
    return float(m)

def _team_shock(n: int, strength: float = 0.10) -> np.ndarray:
    """Per-team multiplicative shock shared by teammates to induce correlation."""
    return np.random.normal(loc=1.0, scale=float(strength), size=int(n))

def _game_shock(n: int, strength: float = 0.06) -> np.ndarray:
    """Per-game multiplicative shock shared by both teams to induce bring-back correlation."""
    return np.random.normal(loc=1.0, scale=float(strength), size=int(n))

def _american_odds_to_prob(odds) -> float | None:
    """Convert American odds to probability in [0,1]."""
    try:
        if odds is None or (isinstance(odds, float) and np.isnan(odds)):
            return None
        if isinstance(odds, str):
            s = odds.strip()
            if not s:
                return None
            odds = float(s.replace("+", ""))
        odds = float(odds)
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return (-odds) / ((-odds) + 100.0)
    except Exception:
        return None

def _extract_props(row: pd.Series) -> dict:
    """Return dict of props if present in row."""
    nx = lambda *names: next((row.get(n) for n in names if n in row and not pd.isna(row.get(n))), None)
    props = {}
    props["PASS_YDS"] = nx("PASS_YDS", "PSYD", "O/U PSYD", "pass_yards")
    props["RUSH_YDS"] = nx("RUSH_YDS", "RUYD", "O/U RUYD", "rush_yards")
    props["REC_YDS"]  = nx("REC_YDS",  "REYD", "O/U REYD", "rec_yards")
    props["RECEPT"]   = nx("RECEPT",   "REC",  "O/U REC",   "receptions")
    anytd = nx("ANYTD_PROB", "ANYTD_prob", "ANYTD PROB", "1+ TD", "ANYTD")
    if anytd is not None and isinstance(anytd, (int, float)) and 0 <= float(anytd) <= 1:
        props["ANYTD_PROB"] = float(anytd)
    else:
        p = _american_odds_to_prob(anytd)
        if p is not None:
            props["ANYTD_PROB"] = float(p)
    passtd = nx("PASS_TD_PROB", "PASS_TD", "1+ PASS TD")
    if passtd is not None and isinstance(passtd, (int, float)) and 0 <= float(passtd) <= 1:
        props["PASS_TD_PROB"] = float(passtd)
    return props

def _props_points_estimate(pos: str, props: dict) -> float | None:
    """Convert available props to rough DK points for blending."""
    if not props:
        return None
    if pos == "QB":
        y = props.get("PASS_YDS")
        return 0.04 * float(y) if y is not None else None
    pts = 0.0; got = False
    ry = props.get("RUSH_YDS"); recy = props.get("REC_YDS"); rec = props.get("RECEPT")
    if ry is not None: pts += 0.1 * float(ry); got = True
    if recy is not None: pts += 0.1 * float(recy); got = True
    if rec is not None: pts += 1.0 * float(rec); got = True
    return pts if got else None

def _anytd_lambda(prob: float) -> float:
    """P(k>=1) = 1 - exp(-λ) => λ = -ln(1 - p)."""
    p = float(np.clip(float(prob), 0.0, 0.999999))
    return 0.0 if p <= 0 else -float(np.log(1.0 - p))

def _script_multipliers(pos: str, spread: float | None) -> tuple[float, float]:
    """Return (mean_mult, sd_mult) from spread (favorite negative)."""
    if spread is None or pd.isna(spread): return 1.0, 1.0
    s = float(spread)
    if s <= -7.0:  # heavy favorite
        if pos == "RB": return 1.08, 0.98
        if pos in ("WR", "TE"): return 0.98, 0.98
        if pos == "QB": return 0.99, 0.98
    elif s >= 7.0:  # heavy dog
        if pos == "QB": return 1.05, 1.05
        if pos in ("WR", "TE"): return 1.05, 1.05
        if pos == "RB": return 0.97, 1.02
    return 1.0, 1.0

# -------------------------
# Main entry
# -------------------------

def process_uploaded_file(
    uploaded_df: pd.DataFrame,
    num_sims: int = DEFAULT_SIMS,
    ml_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build per-player FPTS distributions with correlation & props-aware tweaks."""
    df = uploaded_df.copy()

    # Normalize key columns
    for c in ["TEAM", "OPP", "POS", "PLAYER"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            if c in ("TEAM", "OPP", "POS"):
                df[c] = df[c].str.upper()

    # Merge ML quantiles/implied/defense if provided
    if ml_df is not None and len(ml_df):
        md = ml_df.copy()
        for c in ["TEAM", "POS", "PLAYER"]:
            if c in md.columns:
                md[c] = md[c].astype(str).str.strip().str.upper()
        on_cols = ["PLAYER", "TEAM"]
        if "PLAYER" not in df.columns or df["PLAYER"].isna().all():
            on_cols = ["TEAM"]
        df = df.merge(md, on=on_cols, how="left", suffixes=("", "_ML"))

    # Ensure required columns
    if not {"TEAM", "OPP"}.issubset(df.columns):
        raise ValueError("CSV must have TEAM and OPP columns.")

    # Matchups and correlation shocks
    matchups = df[["TEAM", "OPP"]].drop_duplicates().to_records(index=False).tolist()
    team_to_shock = {tuple(m): _team_shock(num_sims, strength=0.10) for m in matchups}
    games = {tuple(sorted(m)) for m in matchups}
    game_to_shock = {g: _game_shock(num_sims, strength=0.06) for g in games}

    out_rows: list[dict] = []

    # Iterate players
    for idx, row in df.iterrows():
        team = row["TEAM"]; opp = row["OPP"]; pos = str(row.get("POS") or "")
        name = str(row.get("PLAYER") or f"{team}_{pos}_{idx}")
        sal = float(row.get("SAL", np.nan)) if "SAL" in row else np.nan

        # Implied & spread if available
        imp = row.get("implied_team_pts")
        if (imp is None or pd.isna(imp)) and "TM_Score" in df.columns:
            imp = row.get("TM_Score")
        spread = None
        for sc in ("SPRD", "Spread", "spread"):
            if sc in df.columns:
                spread = row.get(sc); break

        # ML quantiles preferred
        p10 = row.get("p10"); p50 = row.get("p50"); p90 = row.get("p90")
        used_ml = False; use_split = False
        if not (pd.isna(p10) or pd.isna(p50) or pd.isna(p90)):
            mu, sd_left, sd_right = _splitnorm_params_from_quantiles(float(p10), float(p50), float(p90))
            used_ml = True; use_split = True
            base_mu, base_sd = mu, max((sd_left + sd_right) / 2.0, 1e-6)
        else:
            base = float(row.get("FPTS", np.nan)) if "FPTS" in df.columns else np.nan
            if pd.isna(base) or base <= 0: base = 6.0
            mu, sd = base, max(0.30 * base, 1.50)
            base_mu, base_sd = mu, sd

        # Props blend (optional)
        props = _extract_props(row)
        prop_pts = _props_points_estimate(pos, props) if props else None
        if prop_pts is not None:
            w = 0.30 if pos in ("RB", "QB") else 0.35
            base_mu = (1.0 - w) * base_mu + w * float(prop_pts)

        # Spread -> script multipliers
        mean_mul, sd_mul = _script_multipliers(pos, spread)
        mu = base_mu * mean_mul
        if use_split:
            sd_left = max(sd_left * sd_mul, 1e-6)
            sd_right = max(sd_right * sd_mul, 1e-6)
        else:
            sd = max(base_sd * sd_mul, 1e-6)

        # Environment & defense
        env_mul = _env_multiplier_from_implied(imp, pos)
        def_mul = _defense_multiplier(row)
        mu *= (env_mul * def_mul)
        if use_split:
            sd_left *= (env_mul * def_mul); sd_right *= (env_mul * def_mul)
        else:
            sd *= (env_mul * def_mul)

        # TD modeling
        anytd_prob = props.get("ANYTD_PROB") if props else None
        td_lambda = 0.0; td_points_per = 6.0
        if pos == "QB":
            pass_prob = props.get("PASS_TD_PROB") if props else None
            td_lambda = _anytd_lambda(pass_prob) if (pass_prob is not None) else 0.0
            td_points_per = 4.0
        elif pos in ("RB", "WR", "TE") and (anytd_prob is not None):
            td_lambda = _anytd_lambda(anytd_prob)
            td_points_per = 6.0

        if td_lambda > 0.0:
            non_td_mu = max(mu - td_lambda * td_points_per, 0.0)
            if use_split:
                base_draws = _sample_split_normal(non_td_mu, sd_left, sd_right, num_sims)
            else:
                base_draws = _sample_player_points(non_td_mu, sd, num_sims)
            td_counts = np.random.poisson(lam=float(td_lambda), size=int(num_sims))
            draws = np.clip(base_draws + td_counts * td_points_per, 0.0, 80.0)
        else:
            if use_split:
                draws = _sample_split_normal(mu, sd_left, sd_right, num_sims)
            else:
                draws = _sample_player_points(mu, sd, num_sims)

        # Correlation shocks
        shock_team = team_to_shock.get((team, opp))
        if shock_team is not None:
            draws = np.clip(draws * shock_team, 0.0, 80.0)
        gkey = tuple(sorted((team, opp)))
        shock_game = game_to_shock.get(gkey)
        if shock_game is not None:
            draws = np.clip(draws * shock_game, 0.0, 80.0)

        # Summaries
        mean = float(np.mean(draws)); std = float(np.std(draws, ddof=0))
        p10o = float(np.percentile(draws, 10)); p25o = float(np.percentile(draws, 25))
        p50o = float(np.percentile(draws, 50)); p75o = float(np.percentile(draws, 75))
        p90o = float(np.percentile(draws, 90))
        boom_thr = max(30.0, (sal/1000.0)*4.0) if not pd.isna(sal) else (mean + std)
        bust_thr = 8.0
        boom_pct = float((draws >= boom_thr).mean() * 100.0)
        bust_pct = float((draws <= bust_thr).mean() * 100.0)
        z_ceiling = float((p90o - p50o) / (std + 1e-6))

        used_sources = ";".join([src for src in [
            ("ML" if used_ml else ""),
            ("PROPS" if prop_pts is not None else ""),
            ("ENV" if (imp is not None and not pd.isna(imp)) else ""),
            ("DEF" if (not pd.isna(row.get("pass_def_mult", np.nan)) or not pd.isna(row.get("rush_def_mult", np.nan))) else ""),
            ("SPREAD" if (spread is not None and not pd.isna(spread)) else ""),
        ] if src])

        out_rows.append({
            "PLAYER": name, "TEAM": team, "OPP": opp, "POS": pos, "SAL": sal,
            "SIMS": int(num_sims),
            "FPTS_mean": round(mean, 2),
            "FPTS_p10": round(p10o, 2), "FPTS_p25": round(p25o, 2), "FPTS_p50": round(p50o, 2),
            "FPTS_p75": round(p75o, 2), "FPTS_p90": round(p90o, 2),
            "FPTS_std": round(std, 2),
            "Boom%": round(boom_pct, 1), "Bust%": round(bust_pct, 1), "Z_ceiling": round(z_ceiling, 3),
            "GameKey": f"{gkey[0]}@{gkey[1]}",
            "ITT": imp if (imp is not None and not pd.isna(imp)) else np.nan,
            "Spread": spread if (spread is not None and not pd.isna(spread)) else np.nan,
            "used_ML": used_ml, "UsedSources": used_sources,
        })

    return pd.DataFrame(out_rows)
