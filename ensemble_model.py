# ensemble_model.py
# Defines the ensemble model that combines LightGBM and the Neural Network.

import numpy as np
import pandas as pd
import torch

class EnsembleModel:
    def __init__(self, lgbm_models: dict, nn_model: torch.nn.Module):
        """
        Initializes the ensemble with trained models.
        - lgbm_models: A dictionary of your 3 trained LightGBM quantile models.
        - nn_model: Your trained QuantileNN from nfl_neural_net.py.
        """
        self.lgbm_models = lgbm_models
        self.nn_model = nn_model
        if self.nn_model:
            self.nn_model.eval() # Set the neural network to evaluation mode

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates final predictions by averaging the outputs of the base models.
        Returns three numpy arrays: (p10, p50, p90) predictions.
        """
        # 1. Get predictions from LightGBM models
        lgbm_p10 = self.lgbm_models[0.10].predict(X)
        lgbm_p50 = self.lgbm_models[0.50].predict(X)
        lgbm_p90 = self.lgbm_models[0.90].predict(X)

        # 2. Get predictions from Neural Network
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        with torch.no_grad():
            nn_preds_tensor = self.nn_model(X_tensor)
        nn_preds = nn_preds_tensor.numpy()
        nn_p10, nn_p50, nn_p90 = nn_preds[:, 0], nn_preds[:, 1], nn_preds[:, 2]

        # 3. Average the predictions for the final ensemble result
        avg_p10 = (lgbm_p10 + nn_p10) / 2.0
        avg_p50 = (lgbm_p50 + nn_p50) / 2.0
        avg_p90 = (lgbm_p90 + nn_p90) / 2.0

        return avg_p10, avg_p50, avg_p90