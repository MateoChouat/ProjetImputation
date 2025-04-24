# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:04:03 2025

@author: mc
"""

import torch.nn as nn
import torch

class ImputationModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(hidden_size, 1)
        self.fc_logvar = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.linear1(x)
        h = self.relu(h)
        h = self.norm(h)
        h = self.dropout(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        output = mu + eps * std  # échantillon

        return output, mu, logvar