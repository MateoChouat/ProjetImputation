# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:39:17 2025

@author: mc
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class MCDropoutNet(nn.Module):
    def __init__(self, input_dim, hidden_size=128, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.drop1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = nn.Dropout(p=dropout)
        self.out_mean = nn.Linear(hidden_size, 1)  # Prédiction moyenne
        self.out_logvar = nn.Linear(hidden_size, 1)  # Logarithme de la variance

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        mean = self.out_mean(x)
        logvar = self.out_logvar(x)  # Estimation de la variance logarithmique
        return mean#, logvar  # Retourner les deux valeurs

def predict_mc(model, x, n_samples=100):
    preds = torch.stack([model(x).squeeze(-1) for _ in range(n_samples)])  # Suppression des dimensions inutiles
    mean = preds.mean(dim=0)  # Moyenne sur les échantillons MC
    std = preds.std(dim=0)  # Écart-type
    return mean, std