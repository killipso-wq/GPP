# env_modifiers.py
# Permanent environment nudges: dome/retractable venues with tiny, capped effects.

from typing import Optional

# Fixed domes
DOME_TEAMS = {"ARI", "DET", "MIN", "NO", "LV"}
# Retractable roofs
RETRACTABLE_TEAMS = {"ARI", "ATL", "DAL", "IND", "HOU"}

ALL_INDOOR = DOME_TEAMS | RETRACTABLE_TEAMS


def is_dome_game(team: Optional[str], opp: Optional[str], is_home: Optional[bool] = None) -> bool:
    """
    Return True if the game is played in an indoor venue (dome or retractable roof).
    If is_home is known:
        - Use the home team’s venue.
    If is_home is None:
        - Treat as indoor if either team plays in an indoor venue (both benefit).
    """
    t = (team or "").upper().strip()
    o = (opp or "").upper().strip()

    if is_home is True:
        return t in ALL_INDOOR
    if is_home is False:
        return o in ALL_INDOOR
    # Unknown home/away – assume indoor if either side is an indoor team
    return (t in ALL_INDOOR) or (o in ALL_INDOOR)


def dome_pos_shift(pos: Optional[str]) -> float:
    """
    Tiny, position-specific median boost indoors (kept small & realistic).
    """
    p = (pos or "").upper().strip()
    if p in ("QB", "WR"):
        return 0.02    # +2%
    if p == "TE":
        return 0.015   # +1.5%
    if p == "RB":
        return 0.005   # +0.5%
    return 0.0


def _parse_home_flag(val) -> Optional[bool]:
    """
    Try to parse a 'Home?' style field into True/False. Returns None if unknown.
    """
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return True
    if s in ("no", "n", "false", "0"):
        return False
    return None


def apply_dome_and_env(row, med_base: float, p90_base: float, env_shift_other: float = 0.0):
    """
    Given a row (with TEAM, OPP, POS, optional 'Home?'), a base median (med_base),
    and base p90 (p90_base), return (med_final, p90_final) after:
      • adding a tiny indoor (dome/roof) bump by position,
      • combining with any existing environment shift you already computed,
      • capping total shift at ±10%,
      • adding a tiny +1% upside to p90 for QB/WR indoors.

    This is intentionally conservative so it won’t blow up results.
    """
    team = row.get("TEAM") or row.get("team")
    opp  = row.get("OPP")  or row.get("opp")
    pos  = row.get("POS")  or row.get("pos")
    home_flag = _parse_home_flag(row.get("Home?") or row.get("home"))

    env_shift = float(env_shift_other or 0.0)

    if is_dome_game(team, opp, home_flag):
        env_shift += dome_pos_shift(pos)
        # a touch more upside for pass game indoors
        if (pos or "").upper() in ("QB", "WR"):
            p90_base = p90_base * 1.01  # +1% to p90 only

    # cap total environment move to ±10%
    env_shift = max(-0.10, min(0.10, env_shift))

    med_final = med_base * (1.0 + env_shift)
    return med_final, p90_base
