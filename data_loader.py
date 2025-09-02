import os, pandas as pd
ROOT = os.path.join(os.path.dirname(__file__), "data_cache", "parquet")

def load_cached(name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(ROOT, f"{name}.parquet"))

def get_all_cached():
    return {
        "weekly": load_cached("weekly"),
        "pbp": load_cached("pbp"),
        "rosters": load_cached("rosters"),
        "snap_counts": load_cached("snap_counts"),
        "ngs_receiving": load_cached("ngs_receiving"),
        "ngs_rushing": load_cached("ngs_rushing"),
        "ngs_passing": load_cached("ngs_passing"),
        "team_off_def_combined": load_cached("team_off_def_combined"),
    }
