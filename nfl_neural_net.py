# nfl_neural_net.py
# Defines the PyTorch Neural Network for quantile regression.

import torch
import torch.nn as nn

# The quantiles we want the model to predict (p10, p50, p90)
QUANTILES = [0.10, 0.50, 0.90]

class QuantileLoss(nn.Module):
    """
    This is the Pinball loss function. It's the key to making a neural network
    predict quantiles instead of just a single average value.
    """
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            # This is the pinball loss formula
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        
        # Average the loss across all quantiles and all samples in the batch
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

class QuantileNN(nn.Module):
    """
    A simple but effective Multi-Layer Perceptron (MLP) for quantile regression.
    - input_dim: The number of features in your dataset (e.g., SAL, OWN, defensive stats, etc.).
    - output_dim: Must be 3, for our three quantiles (p10, p50, p90).
    """
    def __init__(self, input_dim: int, output_dim: int = len(QUANTILES)):
        super(QuantileNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), # Helps stabilize training
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)