import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import optuna
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from models.mc_dropout import MCDropoutNet, predict_mc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gaussian_nll(y_pred, logvar, y_true):
    if logvar is None:
        logvar = torch.zeros_like(y_pred)
    logvar = torch.clamp(logvar, min=-10, max=10)
    loss = 0.5 * torch.exp(-logvar) * (y_true - y_pred)**2 + 0.5 * logvar
    return loss.mean()

def train_mc_model(X_train, y_train, X_val, y_val, X_missing):
    """
    Entraîne un modèle MC Dropout et retourne les prédictions pour X_missing.
    """
    def objective_mc(trial):
        hidden_size = trial.suggest_int("hidden_size", 32, 256)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        model = MCDropoutNet(input_dim=X_train.shape[1],
                             hidden_size=hidden_size, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = gaussian_nll

        best_loss = float('inf')
        patience = 5  # Réduction de la patience pour accélérer l'arrêt précoce
        epochs_no_improve = 0

        # Réduire le nombre d'époques pour l'optimisation : par exemple 50
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            if isinstance(outputs, tuple):
                y_pred = outputs[0]
                logvar = outputs[1] if len(outputs) >= 2 else None
            else:
                y_pred = outputs
                logvar = None
            loss = criterion(y_pred, logvar, y_train)
            loss.backward()
            optimizer.step()

            # Validation rapide avec moins d'échantillons MC pour accélérer
            model.eval()
            with torch.no_grad():
                val_pred, _ = predict_mc(model, X_val, n_samples=100)  # Utiliser 100 échantillons pendant l'optim
                val_loss = mean_absolute_error(val_pred.cpu().numpy(), y_val.cpu().numpy())
            
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
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
    study_mc.optimize(objective_mc, n_trials=30)  # Réduire le nombre de trials pour accélérer

    # Entraînement final avec les meilleurs hyperparamètres trouvés
    best_params = study_mc.best_params
    final_model = MCDropoutNet(input_dim=X_train.shape[1],
                               hidden_size=best_params['hidden_size'],
                               dropout=best_params['dropout']).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])

    # Vous pouvez également réduire le nombre d'époques de l'entraînement final, par exemple 150 au lieu de 300
    for epoch in tqdm(range(150), desc="Entraînement final"):
        final_model.train()
        optimizer.zero_grad()
        outputs = final_model(X_train)
        if isinstance(outputs, tuple):
            y_pred = outputs[0]
            logvar = outputs[1] if len(outputs) >= 2 else None
        else:
            y_pred = outputs
            logvar = None
        loss = gaussian_nll(y_pred, logvar, y_train)
        loss.backward()
        optimizer.step()

    # Prédictions finales pour X_missing avec moins d'échantillons MC (ex. 500)
    final_model.eval()
    with torch.no_grad():
        pred_mean, pred_std = predict_mc(final_model, X_missing, n_samples=500)

    pred_mean = torch.clamp(pred_mean, min=1880, max=1980)
    pred_mean_np = pred_mean.cpu().numpy()
    pred_std_np = pred_std.cpu().numpy()

    # Vérification de cohérence
    incoherences = []
    for i, pred in enumerate(pred_mean_np):
        if pred < 1880 or pred > 1980:
            incoherences.append(f"Index {i}: Prédiction hors des bornes ({pred})")
    if incoherences:
        print("=== Incohérences détectées ===")
        for incoherence in incoherences:
            print(incoherence)

    return final_model, pred_mean_np, pred_std_np

def train_mc_model_with_cv(X, y, X_missing, n_splits=5):
    """
    Effectue l'entraînement avec validation croisée et retourne les prédictions finales pour X_missing.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    pred_means = []
    pred_stds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f'=== Fold {fold + 1}/{n_splits} ===')
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        final_model, pred_mean, pred_std = train_mc_model(X_train, y_train, X_val, y_val, X_missing)

        mae = mean_absolute_error(pred_mean, y_val.cpu().numpy())
        mse = mean_squared_error(pred_mean, y_val.cpu().numpy())
        print(f"MAE pour le Fold {fold + 1}: {mae}")
        print(f"MSE pour le Fold {fold + 1}: {mse}")

        fold_results.append((mae, mse))
        pred_means.append(pred_mean)
        pred_stds.append(pred_std)

    mean_mae = np.mean([r[0] for r in fold_results])
    mean_mse = np.mean([r[1] for r in fold_results])
    print(f"MAE moyen sur tous les folds: {mean_mae}")
    print(f"MSE moyen sur tous les folds: {mean_mse}")

    final_pred_mean = np.mean(pred_means, axis=0)
    final_pred_std = np.mean(pred_stds, axis=0)

    plt.errorbar(range(len(final_pred_mean)), final_pred_mean, yerr=final_pred_std,
                 fmt='o', label="Predictions avec incertitudes")
    plt.xlabel("Index")
    plt.ylabel("Valeur prédite")
    plt.legend()
    plt.title("Prédictions avec intervalles de confiance")
    plt.show()

    return final_model, final_pred_mean, final_pred_std

if __name__ == "__main__":
    # Exemple de données fictives
    X = torch.randn(100, 10).to(device)   # 100 exemples, 10 caractéristiques
    y = torch.randn(100).to(device)        # 100 cibles
    X_missing = torch.randn(10, 10).to(device)  # 10 exemples à prédire

    model, pred_mean, pred_std = train_mc_model_with_cv(X, y, X_missing, n_splits=5)

    print("=== Résultats finaux ===")
    print(f"Prédictions moyennes pour X_missing : {pred_mean}")
    print(f"Incertitudes pour X_missing : {pred_std}")
