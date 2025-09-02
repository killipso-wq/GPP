# prepare_training_data.py
# This script loads raw weekly NFL data, engineers features, and saves the
# final model-ready 'historical_data.csv' file.
# VERSION 5: Added reset_index() to definitively fix "duplicate labels" error.

import nfl_data_py as nfl
import pandas as pd
import numpy as np

print("--- Starting Data Preparation for Ensemble Model ---")

# --- Configuration ---
YEARS = [2021, 2022, 2023] 
OUTPUT_FILE = "historical_data.csv"

# --- 1. Load Raw Weekly Data ---
print(f"Loading weekly data for seasons: {YEARS}...")
try:
    df_raw = nfl.import_weekly_data(YEARS, downcast=True)
    print("Successfully loaded raw weekly data.")
except Exception as e:
    print(f"\nERROR: Failed to load data using nfl-data-py. Make sure it's installed (`pip install nfl-data-py`).")
    print(f"Error details: {e}")
    exit()

# Data Cleaning: Remove duplicate entries for the same player in the same week.
print(f"Raw data contains {len(df_raw)} rows.")
df_raw = df_raw.drop_duplicates(subset=['player_id', 'season', 'week'], keep='last')
print(f"After removing duplicates, data contains {len(df_raw)} rows.")

# --- 2. Calculate the Target Variable (Actual Fantasy Points) ---
print("Calculating actual fantasy points (our target variable)...")

potential_stat_cols = [
    'passing_yards', 'passing_tds', 'interceptions', 'rushing_yards', 'rushing_tds', 
    'receptions', 'receiving_yards', 'receiving_tds', 'fumbles_lost'
]
existing_stat_cols = [col for col in potential_stat_cols if col in df_raw.columns]
df_raw[existing_stat_cols] = df_raw[existing_stat_cols].fillna(0)

df_raw['actual_fantasy_points'] = (
    df_raw.get('passing_yards', 0) * 0.04 +
    df_raw.get('passing_tds', 0) * 4.0 +
    df_raw.get('interceptions', 0) * -1.0 +
    df_raw.get('rushing_yards', 0) * 0.10 +
    df_raw.get('rushing_tds', 0) * 6.0 +
    df_raw.get('receptions', 0) * 1.0 +
    df_raw.get('receiving_yards', 0) * 0.10 +
    df_raw.get('receiving_tds', 0) * 6.0 +
    df_raw.get('fumbles_lost', 0) * -2.0
)

# --- 3. Engineer Features ---
print("Engineering features (rolling averages, opponent strength)...")

desired_cols = [
    'player_id', 'player_display_name', 'season', 'week', 'position',
    'recent_team', 'opponent_team', 'actual_fantasy_points',
    'receptions', 'targets', 'rushing_attempts', 'passing_attempts',
    'passing_yards', 'rushing_yards', 'receiving_yards'
]
existing_cols = [col for col in desired_cols if col in df_raw.columns]
df = df_raw[existing_cols].copy()

df = df[df['position'].isin(['QB', 'RB', 'WR', 'TE'])]

# --- GEMINI MODIFICATION V5 START ---
# To definitively solve the "duplicate labels" error, we will reset the DataFrame's
# index right before the loop. This ensures pandas has a clean, unique index 
# (0, 1, 2, ...) to work with for the rolling average calculations.
df = df.reset_index(drop=True)
# --- GEMINI MODIFICATION V5 END ---

# A) Rolling Average Features (Player's Recent Performance)
potential_rolling_features = ['receptions', 'targets', 'rushing_attempts', 'actual_fantasy_points']
rolling_features = [feat for feat in potential_rolling_features if feat in df.columns]
print(f"Creating rolling average features for: {rolling_features}")

df_grouped = df.groupby('player_id')
for feature in rolling_features:
    # This operation should now succeed because the df index is clean.
    rolling_mean = df_grouped[feature].rolling(window=4, min_periods=1).mean()
    df[f'rolling_avg_{feature}'] = rolling_mean.reset_index(level=0, drop=True).shift(1)

# B) Opponent Strength Features (Matchup)
team_points_allowed = df.groupby(['opponent_team', 'position'])['actual_fantasy_points'].mean().reset_index()
team_points_allowed = team_points_allowed.rename(columns={
    'actual_fantasy_points': 'avg_points_allowed_to_pos',
    'opponent_team': 'team'
})
df = pd.merge(
    df,
    team_points_allowed,
    how='left',
    left_on=['opponent_team', 'position'],
    right_on=['team', 'position']
)

# --- 4. Finalize the Dataset ---
print("Finalizing the dataset for training...")

final_features = []
for feat in rolling_features:
    final_features.append(f'rolling_avg_{feat}')
if 'avg_points_allowed_to_pos' in df.columns:
    final_features.append('avg_points_allowed_to_pos')

final_target = 'actual_fantasy_points'
final_features = [f for f in final_features if f in df.columns]
df_final = df[final_features + [final_target]].copy()
df_final = df_final.dropna()

print(f"Created final dataset with {len(df_final)} samples and {len(final_features)} features.")

# --- 5. Save to CSV ---
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Success! The training data has been saved to '{OUTPUT_FILE}'")