# optimizer_gpp.py
# VERSION 6 — robust POS normalization (handles "RB/FLEX" etc), safer empty-return, DST handling, debug readout

from __future__ import annotations
import argparse, os, random, csv, sys, re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np

# =====================
# Config + RuleBook
# =====================
@dataclass
class RuleBook:
    site: str = "DK"
    salary_cap: int = 50000
    slots: List[str] = field(default_factory=lambda: ["QB","RB","RB","WR","WR","WR","TE","FLEX","DST"])
    flex_pool: Set[str] = field(default_factory=lambda: {"RB","WR","TE"})
    qb_teammate_min: int = 1          # QB must stack with at least this many WR/TE teammates
    qb_bringback_min: int = 0         # Opponent bringbacks required (WR/TE)
    max_exposure: float = 0.60        # per-player max exposure across portfolio
    max_per_team: int = 4             # max players per NFL team
    min_salary_used: int = 49500      # minimum total salary for a lineup
    min_uniques: int = 2              # distinct-player minimum between any two lineups
    temperature: float = 0.10         # randomness for candidate selection
    seed: int = 7
    col_score: str = "SCORE_gpp"      # primary ranking signal
    lock_list: Set[str] = field(default_factory=set)
    exclude_list: Set[str] = field(default_factory=set)

# ---------- Helpers ----------
def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _upper(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper()

def _empty_float_series() -> pd.Series:
    return pd.Series([], dtype="float64")

def _normalize_positions(df: pd.DataFrame) -> None:
    """
    Normalize position labels:
      - Map common DST aliases to 'DST'
      - Convert variants like 'RB/FLEX', 'WR/FLEX', 'TE/FLEX' -> base 'RB','WR','TE'
      - If unknown but contains RB/WR/TE substrings, map to that base
    """
    if "POS" not in df.columns:
        return
    pos = df["POS"].astype(str).str.upper().str.strip()

    # Map DST aliases
    dst_aliases = {
        "D", "D/ST", "DEF", "DEFENSE", "DEFENSE/SPECIAL TEAMS", "DEF/ST", "DST"
    }
    pos = pos.replace({a: "DST" for a in dst_aliases})

    # Strip FLEX suffixes like RB/FLEX -> RB
    pos = pos.str.replace(r"\s*/\s*FLEX$", "", regex=True)

    # If still composite like "RB/WR", take the first token if it's a known base
    def _first_known(p: str) -> str:
        tokens = re.split(r"[^\w]+", p)
        for t in tokens:
            if t in {"QB","RB","WR","TE","DST"}:
                return t
        # heuristic: look for substrings
        if "QB" in p: return "QB"
        if "RB" in p: return "RB"
        if "WR" in p: return "WR"
        if "TE" in p: return "TE"
        if "DST" in p or "DEF" in p: return "DST"
        return p  # fallback

    pos = pos.apply(_first_known)
    df["POS"] = pos

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure required optimizer columns exist by mapping from common app outputs.
    - OWN       <= RST% (ownership) or 'OWN' if present (0..1)
    - MED_final <= SIM_P50 or PROJ_ENV or PROJ_BASE
    - FPTS_p90  <= SIM_P90 or 1.25 * MED_final (fallback)
    - SCORE_gpp <= if present; else a composite from ENV/MED/CEIL/Boom%
    """
    out = df.copy()

    # Standardize key IDs first
    if "PLAYER" not in out.columns:
        for alt in ["Name","player","Player"]:
            if alt in out.columns:
                out = out.rename(columns={alt: "PLAYER"})
                break

    if "TEAM" not in out.columns:
        for alt in ["Tm","team","TEAM"]:
            if alt in out.columns:
                out = out.rename(columns={alt: "TEAM"})
                break

    if "POS" not in out.columns and "Position" in out.columns:
        out = out.rename(columns={"Position": "POS"})

    if "SAL" not in out.columns:
        for alt in ["Salary","DK_SAL","SALARY","salary","DK Salary"]:
            if alt in out.columns:
                out = out.rename(columns={alt: "SAL"})
                break

    _normalize_positions(out)

    # Ownership
    if "OWN" not in out.columns:
        if "RST%" in out.columns:
            own = out["RST%"].astype(str).str.replace("%","", regex=False)
            out["OWN"] = _coerce_num(own) / 100.0
        elif "OWNERSHIP" in out.columns:
            own = out["OWNERSHIP"].astype(str).str.replace("%","", regex=False)
            out["OWN"] = _coerce_num(own) / 100.0
        else:
            out["OWN"] = 0.0

    # Medians / Ceilings
    if "MED_final" not in out.columns:
        if "SIM_P50" in out.columns:
            out["MED_final"] = _coerce_num(out["SIM_P50"])
        elif "PROJ_ENV" in out.columns:
            out["MED_final"] = _coerce_num(out["PROJ_ENV"])
        else:
            out["MED_final"] = _coerce_num(out.get("PROJ_BASE", pd.Series(0.0, index=out.index)))

    if "FPTS_p90" not in out.columns:
        if "SIM_P90" in out.columns:
            out["FPTS_p90"] = _coerce_num(out["SIM_P90"])
        else:
            base = out["MED_final"].fillna(0.0)
            out["FPTS_p90"] = (base * 1.25).round(2)

    # Score (final GPP score)
    if "SCORE_gpp" not in out.columns:
        boom = _coerce_num(out.get("Boom%", pd.Series(np.nan, index=out.index))).fillna(0.0) / 100.0
        env  = _coerce_num(out.get("PROJ_ENV", out.get("PROJ_BASE", pd.Series(0.0, index=out.index)))).fillna(0.0)
        med  = _coerce_num(out["MED_final"]).fillna(0.0)
        ceil = _coerce_num(out["FPTS_p90"]).fillna(0.0)
        out["SCORE_gpp"] = (0.55*env + 0.35*med + 0.10*ceil) * (1.0 + 0.15*boom)

    # Salary numeric
    out["SAL"] = _coerce_num(out.get("SAL", pd.Series(np.nan, index=out.index)))

    # IDs nice & uppercase
    if "TEAM" in out.columns:
        out["TEAM"] = _upper(out["TEAM"])

    # Ensure OPP exists
    if "OPP" not in out.columns:
        out["OPP"] = ""

    return out

# =====================
# Data loading
# =====================
def load_final_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    df = _map_columns(df)

    # Only coerce numeric on numeric fields
    for c in ["SAL","SCORE_gpp","OWN","MED_final","FPTS_p90"]:
        if c in df.columns:
            df[c] = _coerce_num(df[c])

    # Basic required fields check (after mapping)
    need = ["PLAYER","TEAM","OPP","POS","SAL","SCORE_gpp","OWN","MED_final","FPTS_p90"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    # Drop rows missing essentials
    df = df.dropna(subset=["PLAYER","TEAM","POS","SAL","SCORE_gpp"]).copy()

    # Ranking signal for candidate selection
    df["_score_rank"] = df["SCORE_gpp"].fillna(0.0) + 0.03 * df["FPTS_p90"].fillna(0.0)
    return df

# =====================
# Lineup data model
# =====================
@dataclass
class Lineup:
    idxs: List[int]
    salary: int

    def to_row(self, df: pd.DataFrame) -> Dict[str, object]:
        lp = df.loc[self.idxs]
        p_cols = {f"P{i+1}": name for i, name in enumerate(lp['PLAYER'])}
        pos_cols = {f"Pos{i+1}": pos for i, pos in enumerate(lp['POS'])}
        return {
            **p_cols, **pos_cols,
            "TotalSalary": int(self.salary),
            "TotalScore": round(float(lp["SCORE_gpp"].sum()), 2),
            "TotalMedian": round(float(lp["MED_final"].sum()), 2),
            "TotalCeiling": round(float(lp["FPTS_p90"].sum()), 2),
        }

# ---------- Validation & Constraints ----------
def is_valid_roster(lineup_players: pd.DataFrame, rules: RuleBook) -> bool:
    pos_counts = lineup_players['POS'].value_counts().to_dict()
    # mandatory counts by dedicated slots
    for pos in ["QB","RB","WR","TE","DST"]:
        required = rules.slots.count(pos)
        available = pos_counts.get(pos, 0)
        if available < required:
            return False
        pos_counts[pos] = available - required
    # fill FLEX
    flex_needed = rules.slots.count("FLEX")
    flex_filled = sum(min(pos_counts.get(p, 0), flex_needed) for p in rules.flex_pool)
    return flex_filled >= flex_needed

def satisfies_all_constraints(lineup: Lineup, df: pd.DataFrame, rules: RuleBook) -> bool:
    lp = df.loc[lineup.idxs]
    if not is_valid_roster(lp, rules):
        return False
    if lp["TEAM"].value_counts().max() > rules.max_per_team:
        return False
    if lineup.salary < rules.min_salary_used:
        return False
    qb = lp[lp['POS']=="QB"]
    if qb.empty: 
        return False
    qb_team = qb.iloc[0]["TEAM"]; qb_opp = qb.iloc[0]["OPP"]
    teammates = len(lp[(lp["TEAM"]==qb_team) & (lp["POS"].isin(["WR","TE"]))])
    if teammates < rules.qb_teammate_min: 
        return False
    bringbacks = len(lp[(lp["TEAM"]==qb_opp) & (lp["POS"].isin(["WR","TE"]))])
    if bringbacks < rules.qb_bringback_min: 
        return False
    return True

# ---------- Pools / Builder ----------
def build_slot_pools(df: pd.DataFrame, rules: RuleBook) -> Dict[str, List[int]]:
    pools = {}
    for slot in list(dict.fromkeys(rules.slots)):  # de-dupe while keeping order
        eligible = (df["POS"] == slot) if slot != "FLEX" else df["POS"].isin(rules.flex_pool)
        pool_idxs = df.index[eligible].tolist()
        pool_idxs = [i for i in pool_idxs if df.loc[i, "PLAYER"] not in rules.exclude_list]
        pools[slot] = sorted(pool_idxs, key=lambda i: df.loc[i, "_score_rank"], reverse=True)
    return pools

def pick_candidate(df, pool, used, exposures, built, rules, salary_remaining, slots_left):
    avg_budget = max(1, int(salary_remaining) // max(1, slots_left))
    cand_idxs = [i for i in pool
                 if i not in used
                 and (built == 0 or exposures.get(i,0)/built < rules.max_exposure)
                 and (df.loc[i,"SAL"] <= avg_budget + 2000 or random.random() < 0.30)]
    if not cand_idxs:
        return None
    scores = df.loc[cand_idxs, "_score_rank"].to_numpy(dtype=float)
    own = df.loc[cand_idxs, "OWN"].to_numpy(dtype=float)
    std = float(scores.std()) if len(scores) > 1 else 0.0
    noise = np.random.normal(0.0, rules.temperature * (std if std > 0 else 1.0), size=scores.shape)
    weights = np.maximum(1e-6, scores + noise) * (1.0 - 0.25*np.clip(own,0,1))
    wsum = float(weights.sum())
    if wsum <= 0:
        return None
    return int(np.random.choice(cand_idxs, p=weights/wsum))

def build_lineups(df: pd.DataFrame, rules: RuleBook, n_lineups: int) -> List[Lineup]:
    random.seed(rules.seed); np.random.seed(rules.seed)

    # Auto-handle missing DST: if your slate has zero DST rows, remove the slot and note it
    if "DST" in rules.slots and (df["POS"]=="DST").sum() < 1:
        rules.slots = [s for s in rules.slots if s != "DST"]
        print("[optimizer] No DST rows detected — removed DST slot for this run.", file=sys.stderr)

    slot_pools = build_slot_pools(df, rules)
    exposures = {i: 0 for i in df.index}
    built, seen = [], set()

    def unique_ok(new_idxs, built_lineups):
        if rules.min_uniques <= 1: 
            return True
        new_set = set(new_idxs)
        for L in built_lineups:
            if len(new_set.intersection(set(L.idxs))) >= len(new_set) - rules.min_uniques + 1:
                return False
        return True

    max_tries = n_lineups * 500
    for _ in range(max_tries):
        if len(built) >= n_lineups: 
            break
        idxs, used, salary = [], set(), 0

        # Place locks first (respect slot where possible; if not, try FLEX)
        locked_idx = df[df["PLAYER"].isin(rules.lock_list)].index.tolist()
        current_slots = list(rules.slots)
        for i in locked_idx:
            pos = df.loc[i,"POS"]
            if pos in current_slots:
                idxs.append(i); used.add(i); salary += int(df.loc[i,"SAL"])
                current_slots.remove(pos)
            elif "FLEX" in current_slots and pos in rules.flex_pool:
                idxs.append(i); used.add(i); salary += int(df.loc[i,"SAL"])
                current_slots.remove("FLEX")

        ok = True
        for slot_i, slot in enumerate(current_slots):
            cand = pick_candidate(df, slot_pools.get(slot, []), used, exposures, len(built),
                                  rules, rules.salary_cap - salary, len(current_slots)-slot_i)
            if cand is None:
                ok = False; break
            idxs.append(cand); used.add(cand); salary += int(df.loc[cand,"SAL"])

        if not ok or salary > rules.salary_cap:
            continue
        key = tuple(sorted(idxs))
        if key in seen:
            continue

        lineup = Lineup(idxs=idxs, salary=int(salary))
        if satisfies_all_constraints(lineup, df, rules) and unique_ok(lineup.idxs, built):
            built.append(lineup); seen.add(key)
            for i in idxs: exposures[i] = exposures.get(i,0) + 1

    return built

def export_to_dk_csv(lineups, df, path):
    dk_order = ["QB","RB","RB","WR","WR","WR","TE","FLEX","DST"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(dk_order)
        for lu in lineups:
            roster_map = {pos: [] for pos in ["QB","RB","WR","TE","DST","FLEX"]}
            lp = df.loc[lu.idxs]; placed_ids = []
            # Fill fixed slots first
            for pos in ["QB","TE","DST"]:
                need = dk_order.count(pos)
                for idx, p in lp[lp["POS"]==pos].iterrows():
                    if len(roster_map[pos]) < need:
                        roster_map[pos].append(p["PLAYER"]); placed_ids.append(idx)
            # Then RB/WR
            for pos in ["RB","WR"]:
                need = dk_order.count(pos)
                for idx, p in lp[lp["POS"]==pos].iterrows():
                    if len(roster_map[pos]) < need:
                        roster_map[pos].append(p["PLAYER"]); placed_ids.append(idx)
            # Flex with remaining eligible
            flex_cands = lp[~lp.index.isin(placed_ids)]
            for idx, p in flex_cands.iterrows():
                if p["POS"] in {"RB","WR","TE"} and len(roster_map["FLEX"]) < dk_order.count("FLEX"):
                    roster_map["FLEX"].append(p["PLAYER"])
            # If 'DST' slot not actually in rules (removed earlier), pad with empty cell
            if "DST" not in [s for s in RuleBook().slots]:
                while len(roster_map["DST"]) < dk_order.count("DST"):
                    roster_map["DST"].append("")
            row = roster_map["QB"] + roster_map["RB"] + roster_map["WR"] + roster_map["TE"] + roster_map["FLEX"] + roster_map["DST"]
            w.writerow(row)

# =====================
# Debug readout
# =====================
def debug_pool_report(df: pd.DataFrame, rules: RuleBook) -> None:
    try:
        print("\n[optimizer] === Player Pool Debug ===", file=sys.stderr)
        # Position counts (after normalization)
        pos_counts = df["POS"].value_counts(dropna=False).sort_index()
        print("[optimizer] Position counts:", file=sys.stderr)
        for pos, cnt in pos_counts.items():
            print(f"  - {pos}: {int(cnt)}", file=sys.stderr)

        # DST presence
        dst_count = int((df["POS"] == "DST").sum())
        print(f"[optimizer] DST rows: {dst_count}", file=sys.stderr)

        # Salary stats
        sal = _coerce_num(df.get("SAL", _empty_float_series())).dropna()
        if not sal.empty:
            print(f"[optimizer] Salary min/median/max: {int(sal.min())} / {int(sal.median())} / {int(sal.max())}", file=sys.stderr)

        # Ownership stats
        own = _coerce_num(df.get("OWN", _empty_float_series())).dropna()
        if not own.empty:
            q = own.quantile([0.1,0.5,0.9])
            print(f"[optimizer] Ownership p10/median/p90: {q.iloc[0]:.3f} / {q.iloc[1]:.3f} / {q.iloc[2]:.3f}", file=sys.stderr)

        # Score stats
        score = _coerce_num(df.get("SCORE_gpp", _empty_float_series())).dropna()
        if not score.empty:
            print(f"[optimizer] SCORE_gpp mean ± std: {score.mean():.2f} ± {score.std():.2f}", file=sys.stderr)

        # Active rules
        print("[optimizer] Rules in effect:", file=sys.stderr)
        print(f"  slots={rules.slots}, cap={rules.salary_cap}, min_salary_used={rules.min_salary_used}, max_per_team={rules.max_per_team}", file=sys.stderr)
        print(f"  stack: teammates>={rules.qb_teammate_min}, bringbacks>={rules.qb_bringback_min}", file=sys.stderr)
        print(f"  max_exposure={rules.max_exposure:.2f}, uniques={rules.min_uniques}, temperature={rules.temperature:.2f}, seed={rules.seed}", file=sys.stderr)
        print("[optimizer] ==========================\n", file=sys.stderr)
    except Exception as e:
        print(f"[optimizer] debug_pool_report error: {e}", file=sys.stderr)

# =====================
# CLI
# =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to final table CSV (from app) or any CSV with mappable columns")
    ap.add_argument("--out", dest="out_csv", default="outputs/lineups.csv")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--cap", type=int, default=50000)
    ap.add_argument("--maxexp", type=float, default=0.60)
    ap.add_argument("--stack", default="1,0", help="QB teammate min,bringback min (e.g., '1,1')")
    ap.add_argument("--minsal", type=int, default=49500)
    ap.add_argument("--uniques", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--locks", default=None, help="Comma-separated PLAYER names to lock")
    ap.add_argument("--xout", default=None, help="Comma-separated PLAYER names to exclude")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    rules = RuleBook()
    rules.salary_cap = args.cap; rules.max_exposure = args.maxexp; rules.min_salary_used = args.minsal
    rules.min_uniques = args.uniques; rules.temperature = args.temp; rules.seed = args.seed
    try:
        mates, bb = args.stack.split(",")
        rules.qb_teammate_min = int(mates); rules.qb_bringback_min = int(bb)
    except Exception:
        pass
    if args.locks: rules.lock_list = set([s.strip() for s in args.locks.split(",") if s.strip()])
    if args.xout:  rules.exclude_list = set([s.strip() for s in args.xout.split(",")  if s.strip()])

    df = load_final_table(args.in_path)
    debug_pool_report(df, rules)

    lineups = build_lineups(df, rules, n_lineups=args.n)
    if not lineups:
        print("No lineups built. Try relaxing constraints or verify inputs.", file=sys.stderr)
        return

    out_rows = [lu.to_row(df) for lu in lineups]
    out_df = pd.DataFrame(out_rows).sort_values("TotalScore", ascending=False).reset_index(drop=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"✓ Wrote {len(out_df)} analysis lineups -> {os.path.abspath(args.out_csv)}")

    dk_upload_path = os.path.join(os.path.dirname(args.out_csv), "dk_upload.csv")
    export_to_dk_csv(lineups, df, dk_upload_path)
    print(f"✓ Wrote {len(lineups)} DK formatted lineups -> {os.path.abspath(dk_upload_path)}")

    # Exposure report
    exposures = {}
    for lu in lineups:
        for i in lu.idxs:
            exposures[i] = exposures.get(i, 0) + 1
    exp_df = pd.DataFrame.from_dict(exposures, orient='index', columns=['Count'])
    exp_df['PLAYER'] = df.loc[exp_df.index, 'PLAYER']
    exp_df['ExposurePct'] = 100 * exp_df['Count'] / len(lineups)
    exp_out = os.path.splitext(args.out_csv)[0] + "_exposure.csv"
    exp_df[['PLAYER','Count','ExposurePct']].sort_values('Count', ascending=False).to_csv(exp_out, index=False)
    print(f"✓ Wrote exposure report -> {os.path.abspath(exp_out)}")

# =====================
# App-callable entrypoint
# =====================
def generate_lineups(df_in: pd.DataFrame, num_lineups: int = 20, rules: Optional[RuleBook] = None) -> pd.DataFrame:
    """
    Call from Streamlit:
        from optimizer_gpp import generate_lineups
        lineups_df = generate_lineups(final_tbl, 20)
    Returns a DataFrame of lineups sorted by TotalScore.
    """
    rules = rules or RuleBook()

    # Accept either a prepared DataFrame or a path to CSV
    if isinstance(df_in, pd.DataFrame):
        df = _map_columns(df_in)
        _normalize_positions(df)
        # Numeric fields
        for c in ["SAL","SCORE_gpp","OWN","MED_final","FPTS_p90"]:
            if c in df.columns:
                df[c] = _coerce_num(df[c])
        # Ranking signal (ensure present in DF path)
        if "_score_rank" not in df.columns:
            df["_score_rank"] = df["SCORE_gpp"].fillna(0.0) + 0.03 * df.get("FPTS_p90", 0.0).fillna(0.0)
    else:
        df = load_final_table(str(df_in))  # already mapped & ranked

    # Drop any rows missing essentials (safety)
    df = df.dropna(subset=["PLAYER","TEAM","POS","SAL","SCORE_gpp"]).copy()

    # Debug readout to stderr (visible in Streamlit terminal; harmless in app)
    debug_pool_report(df, rules)

    # Build the portfolio
    lus = build_lineups(df, rules, n_lineups=int(num_lineups))
    rows = [lu.to_row(df) for lu in lus]

    # Safe return (avoid 'TotalScore' KeyError on empty)
    if not rows:
        return pd.DataFrame(columns=[
            "P1","P2","P3","P4","P5","P6","P7","P8","P9",
            "Pos1","Pos2","Pos3","Pos4","Pos5","Pos6","Pos7","Pos8","Pos9",
            "TotalSalary","TotalScore","TotalMedian","TotalCeiling"
        ])

    return pd.DataFrame(rows).sort_values("TotalScore", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    main()
