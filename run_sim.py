import sys, pandas as pd
from data_loader import get_all_cached
from attributes import calculate_player_attributes
from simulation_engine import process_uploaded_file

def main(in_csv, out_csv="simulation_results.csv"):
    hist = get_all_cached()
    attrs = calculate_player_attributes(hist)
    df = pd.read_csv(in_csv)
    out = process_uploaded_file(df, hist, attrs)
    out.to_csv(out_csv, index=False)
    print(f"✓ Wrote {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_sim.py path\\to\\players_or_matchups.csv [out.csv]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "simulation_results.csv")
