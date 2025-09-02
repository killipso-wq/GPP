# attributes.py  (robust to schema differences in nfl-data-py)
import pandas as pd
import numpy as np

# Helper: pick the first column from a candidate list that exists in df
def _pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# Helper: build a stable "pid" when no real id is shared (uses name+team)
def _mk_pid_from_name_team(df, name_col, team_col):
    def canon(x):
        if pd.isna(x): return ""
        return str(x).strip().upper()
    return (df[name_col].map(canon) + "|" + df[team_col].map(canon)).replace({"|": pd.NA})

def calculate_player_attributes(historical):
    rosters = historical["rosters"].copy()
    snaps   = historical["snap_counts"].copy()

    # --- Normalize essential columns in ROSTERS ---
    r_id_col   = _pick(rosters, ["player_id","gsis_id","nfl_id","nflId","pfr_id","esb_id","sportradar_id"])
    r_name_col = _pick(rosters, ["player_name","player","full_name","display_name","name"])
    r_team_col = _pick(rosters, ["team","recent_team","club_code"])
    r_wt_col   = _pick(rosters, ["weight","wt"])

    # If any are missing, create sane fallbacks
    if r_name_col is None:
        # last-ditch: create a synthetic name
        rosters["__name__"] = rosters.get("player", rosters.index).astype(str)
        r_name_col = "__name__"
    if r_team_col is None:
        rosters["__team__"] = rosters.get("team", pd.NA)
        r_team_col = "__team__"

    # Compute a unified pid for rosters
    if r_id_col is not None:
        rosters["pid"] = rosters[r_id_col].astype(str)
    else:
        rosters["pid"] = _mk_pid_from_name_team(rosters, r_name_col, r_team_col)

    # Build base attrs frame
    keep = [col for col in [r_name_col, r_team_col, r_wt_col, "pid", "position"] if col in rosters.columns]
    attrs = rosters[keep].rename(columns={
        r_name_col: "player_name",
        r_team_col: "team",
        r_wt_col: "W",
    }).copy()

    # Drop rows without a pid or team (FAs not useful for matchup sims)
    attrs = attrs.dropna(subset=["pid", "team"]).copy()
    attrs.drop_duplicates(subset=["pid","team"], inplace=True)

    # Ensure required columns exist
    if "position" not in attrs.columns:
        attrs["position"] = pd.NA

    # --- Normalize essential columns in SNAP COUNTS ---
    s_id_col   = _pick(snaps, ["player_id","gsis_id","nfl_id","nflId","pfr_id","esb_id","sportradar_id"])
    s_name_col = _pick(snaps, ["player_name","player","full_name","display_name","name"])
    s_team_col = _pick(snaps, ["team","recent_team","club_code"])
    s_off_col  = _pick(snaps, ["offense_snaps","offense_snaps_played","offense"])
    s_def_col  = _pick(snaps, ["defense_snaps","defense_snaps_played","defense"])

    if snaps is not None and len(snaps) > 0:
        if s_name_col is None:
            snaps["__name__"] = snaps.get("player", snaps.index).astype(str)
            s_name_col = "__name__"
        if s_team_col is None:
            snaps["__team__"] = snaps.get("team", pd.NA)
            s_team_col = "__team__"

        # compute pid for snaps (prefer real id, else name+team)
        if s_id_col is not None:
            snaps["pid"] = snaps[s_id_col].astype(str)
        else:
            snaps["pid"] = _mk_pid_from_name_team(snaps, s_name_col, s_team_col)

        # snap columns present?
        snap_cols = [c for c in [s_off_col, s_def_col] if c is not None]
        if snap_cols:
            agg = snaps.groupby("pid")[snap_cols].sum(numeric_only=True)
            agg["total_snaps"] = agg.sum(axis=1)
            eligible_pids = set(agg[agg["total_snaps"] >= 100].index)
        else:
            # no usable snap columns — treat everyone as eligible
            eligible_pids = set(attrs["pid"])
    else:
        # no snap data at all
        eligible_pids = set(attrs["pid"])

    attrs["is_eligible"] = attrs["pid"].isin(eligible_pids)

    # --- Placeholder physics scalars (safe defaults) ---
    rng = np.random.default_rng(42)
    n = len(attrs)
    attrs["C"]  = rng.normal(15,  2, n)    # capacity-ish
    attrs["S"]  = rng.normal(150, 20, n)   # speed-ish
    attrs["K1"] = rng.normal(2.5, 0.5, n)  # modulator

    # Expose a canonical column name for the rest of the app
    attrs.rename(columns={"pid": "player_id"}, inplace=True)

    # Final column order (only those that exist)
    final_cols = [c for c in ["player_id","player_name","position","team","W","is_eligible","C","S","K1"] if c in attrs.columns]
    return attrs[final_cols]
