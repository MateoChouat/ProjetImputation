# -*- coding: utf-8 -*-
"""
Pipeline pour l'entraînement d'un modèle MC Dropout avec validation croisée.

Créé le : Jeudi 24 Avril 2025
Auteur : Mateo Chouat
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import optuna
from tqdm import tqdm
import numpy as np

from models.mc_dropout import MCDropoutNet, predict_mc

# Configuration du périphérique (CPU ou GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_mc_model(X_train, y_train, X_val, y_val, X_missing):
    """
    Entraîne un modèle MC Dropout et retourne les prédictions pour X_missing.
    """
    def gaussian_nll(mu, logvar, y):
        logvar = torch.clamp(logvar, min=-10, max=10)
        var = torch.exp(logvar)
        return 0.5 * torch.mean(torch.log(2 * torch.pi * var) + (y - mu) ** 2 / var)

    def objective_mc(trial):
        hidden_size = trial.suggest_int("hidden_size", 32, 256)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        
        model = MCDropoutNet(input_dim=X_train.shape[1], hidden_size=hidden_size, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.L1Loss()

        best_loss = float('inf')
        patience = 10
        epochs_no_improve = 0
        
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred, _ = predict_mc(model, X_val)
                val_loss = mean_absolute_error(val_pred.cpu().numpy(), y_val.cpu().numpy())

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        return best_loss

    # Optimisation des hyperparamètres avec Optuna
    study_mc = optuna.create_study(direction="minimize")
    study_mc.optimize(objective_mc, n_trials=20)

    # Entraînement final avec les meilleurs hyperparamètres trouvés
    best_params = study_mc.best_params
    final_model = MCDropoutNet(input_dim=X_train.shape[1], hidden_size=best_params['hidden_size'], dropout=best_params['dropout']).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])

    for epoch in tqdm(range(300), desc="Entraînement final"):
        final_model.train()
        optimizer.zero_grad()
        y_pred = final_model(X_train)
        loss = F.l1_loss(y_pred, y_train)
        loss.backward()
        optimizer.step()

    # Prédictions finales pour X_missing
    final_model.eval()
    with torch.no_grad():
        pred_mean, pred_std = predict_mc(final_model, X_missing, n_samples=400)

    # Conversion des prédictions en numpy
    pred_mean_np = pred_mean.cpu().numpy()
    pred_std_np = pred_std.cpu().numpy()

    return final_model, pred_mean_np, pred_std_np


def train_mc_model_with_cv(X, y, X_missing, n_splits=5):
    """
    Effectue l'entraînement avec validation croisée et retourne les prédictions finales pour X_missing.

    Parameters:
    - X (torch.Tensor): Données d'entrée.
    - y (torch.Tensor): Cibles.
    - X_missing (torch.Tensor): Données à prédire.
    - n_splits (int): Nombre de folds pour la validation croisée.

    Returns:
    - final_model: Le dernier modèle entraîné.
    - pred_mean (np.array): Moyenne des prédictions pour X_missing.
    - pred_std (np.array): Incertitude des prédictions pour X_missing.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    pred_means = []
    pred_stds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f'=== Fold {fold + 1}/{n_splits} ===')

        # Division des données en train/validation
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Entraînement sur ce fold
        final_model, pred_mean, pred_std = train_mc_model(X_train, y_train, X_val, y_val, X_missing)

        # Stockage des résultats
        fold_results.append(mean_absolute_error(pred_mean, y_val.cpu().numpy()))
        pred_means.append(pred_mean)
        pred_stds.append(pred_std)

        print(f"MAE pour le Fold {fold + 1}: {fold_results[-1]}")

    # Résultats globaux
    mean_mae = np.mean(fold_results)
    print(f"MAE moyen sur tous les folds: {mean_mae}")

    # Moyenne des prédictions sur tous les folds
    final_pred_mean = np.mean(pred_means, axis=0)
    final_pred_std = np.mean(pred_stds, axis=0)

    return final_model, final_pred_mean, final_pred_std


if __name__ == "__main__":
    # Exemple de données fictives
    X = torch.randn(100, 10).to(device)  # 100 exemples, 10 caractéristiques
    y = torch.randn(100).to(device)  # 100 cibles
    X_missing = torch.randn(10, 10).to(device)  # 10 exemples à prédire

    # Entraînement avec validation croisée
    model, pred_mean, pred_std = train_mc_model_with_cv(X, y, X_missing, n_splits=5)

    # Résultats finaux
    print("=== Résultats finaux ===")
    print(f"Prédictions moyennes pour X_missing : {pred_mean}")
    print(f"Incertitudes pour X_missing : {pred_std}")