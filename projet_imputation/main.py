# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:28:54 2025

@author: mc
"""
from data_loader import load_data, preprocess_data
from training.train_mc import train_mc_model
from evaluation.coherence import apply_coherence_check
from evaluation.visualizations import plot_predictions
from evaluation.restore_values import retransform
from config import path
import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def main():
    # Load raw data
    print("🚀 Loading data...")
    df_raw = load_data(path)
    
    # Preprocess data
    print("🔄 Preprocessing data...")
    X_train, y_train, X_val, y_val, X_missing, df_missing, year_max, year_min = preprocess_data(df_raw)
    
    # Convert numpy arrays to PyTorch tensors if necessary
    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32).to(device) if isinstance(x, np.ndarray) else x
    
    X_train, y_train = to_tensor(X_train), to_tensor(y_train)
    X_val, y_val = to_tensor(X_val), to_tensor(y_val)
    X_missing = to_tensor(X_missing)
    
    # Train model and generate predictions
    print("🧠 Training the model...")
    model, pred_mean, pred_std = train_mc_model(X_train, y_train, X_val, y_val, X_missing)
    
    # Debug: Analyze predictions before clamping
    print("📊 Debugging Predictions:")
    print("Predictions before clamping:", pred_mean)
    print("Uncertainties:", pred_std)
    
    #X_train, X_val, X_missing = retransform(X_train, X_val, X_missing, scaler)
    
    # Apply coherence checks to predictions
    print("✅ Applying coherence checks...")
    df_results = apply_coherence_check(df_missing, pred_mean, pred_std, year_max, year_min)
    
    # Visualize predictions with uncertainties
    print("📈 Visualizing results...")
    plot_predictions(pred_mean, pred_std)
    
    # Display results
    print("\n=== Final Results ===")
    print(df_results)
    
if __name__ == '__main__':
    main()