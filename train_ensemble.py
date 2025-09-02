# train_ensemble.py
# Run this script to train and save your new ensemble projection model.
# VERSION 2: Updated feature list to match the output of prepare_training_data.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Import the new classes we just defined
from nfl_neural_net import QuantileNN, QuantileLoss, QUANTILES
from ensemble_model import EnsembleModel

print("--- Starting Ensemble Model Training ---")

# --- 1. Load and Prepare Your Historical Data ---
try:
    df = pd.read_csv("historical_data.csv")
    print(f"Successfully loaded historical_data.csv with {len(df)} rows.")
except FileNotFoundError:
    print("\nERROR: `historical_data.csv` not found.")
    print("Please create this file with your historical training data by running `prepare_training_data.py` first.")
    exit()

# --- GEMINI MODIFICATION ---
# Define your features and target here.
# This list now exactly matches the columns created by prepare_training_data.py
features = [
    'rolling_avg_receptions',
    'rolling_avg_targets',
    'rolling_avg_actual_fantasy_points',
    'avg_points_allowed_to_pos'
]
target = 'actual_fantasy_points'
# --- END MODIFICATION ---

# Make sure all required columns exist
required_cols = features + [target]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"\nERROR: The following required columns are missing from your data: {missing_cols}")
    exit()

df = df.dropna(subset=required_cols)
X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

# Scale features for the Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. Train LightGBM Quantile Models ---
print("\n--- Training LightGBM Models ---")
lgbm_models = {}
for q in QUANTILES:
    print(f"Training LightGBM for quantile: {q:.2f}")
    model = lgb.LGBMRegressor(objective='quantile', alpha=q, random_state=42)
    model.fit(X_train, y_train)
    lgbm_models[q] = model
print("LightGBM training complete.")

# --- 3. Train the Quantile Neural Network ---
print("\n--- Training Neural Network ---")
# Convert data to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss, and optimizer
input_dim = X_train.shape[1]
nn_model = QuantileNN(input_dim=input_dim)
loss_fn = QuantileLoss(quantiles=QUANTILES)
optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.005)
num_epochs = 50 # You can increase this for better performance

nn_model.train()
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = nn_model(X_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"NN Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
print("Neural Network training complete.")


# --- 4. Create and Save the Final Ensemble Model ---
print("\n--- Creating and Saving Ensemble Model ---")
final_ensemble_model = EnsembleModel(lgbm_models=lgbm_models, nn_model=nn_model)

# Save the scaler as well, we'll need it for inference
joblib.dump(scaler, 'feature_scaler.joblib')
# Save the final ensemble model object
joblib.dump(final_ensemble_model, 'ensemble_model.joblib')

print("\n✅ Success! The trained ensemble model has been saved to 'ensemble_model.joblib'")
print("The feature scaler has been saved to 'feature_scaler.joblib'")