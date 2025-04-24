# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:28:54 2025

@author: mc
"""
from data_loader import load_data, preprocess_data
from training.train_mc import train_mc_model
from evaluation.coherence import apply_coherence_check
from evaluation.visualizations import plot_predictions
from config import path
import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    df_raw = load_data(path)
    X_train, y_train, X_val, y_val, X_missing, df_missing, year_max, year_min = preprocess_data(df_raw)
    
    if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    if isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    if isinstance(X_val, np.ndarray):
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    if isinstance(y_val, np.ndarray):
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    if isinstance(X_missing, np.ndarray):
        X_missing = torch.tensor(X_missing, dtype=torch.float32).to(device)
    
    model, pred_mean, pred_std = train_mc_model(X_train, y_train, X_val, y_val, X_missing)
    
    df_results = apply_coherence_check(df_missing, pred_mean, pred_std, year_max, year_min)
    plot_predictions(pred_mean, pred_std)
    print(df_results)
    
if __name__=='__main__':
    main()