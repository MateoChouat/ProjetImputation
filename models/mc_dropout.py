# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:39:17 2025

@author: mc
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class MCDropoutNet(nn.Module):
    def __init__(self, input_dim, hidden_size = 128, dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.drop1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = nn.Dropout(p=dropout)
        self.out = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.out(x)
    
    
def predict_mc(model, x, n_samples=30):
    model.train()
    preds = torch.stack([model(x) for _ in range(n_samples)]).squeeze()
    return preds.mean(dim=0), preds.std(dim=0)