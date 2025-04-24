# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:34:49 2025

@author: mc
"""
import torch
import torch.nn.functional as F
from models.mc_dropout import MCDropoutNet, predict_mc
import geopandas as gpd 
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from config import cat_features, num_features, target_column
from sklearn.preprocessing import StandardScaler, PowerTransformer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(path):
    gdf = gpd.read_file(path)
    gdf['NOM_COM'] = gdf['NOM_COM'].fillna("Antibes")
    return gdf

def preprocess_data(gdf):
    colonnes_souhaitees = cat_features + num_features + [target_column]

    df = gdf[colonnes_souhaitees]

    df = pd.get_dummies(df, columns = cat_features, drop_first=True)
    print("l'objet est : ", df['ANNEEPOSE'].dtype)


    df["DIAMETRE_LOG"] = np.log1p(df["DIAMETRE"])
    df["LENGTH_SQRT"] = np.sqrt(df["SHAPE_LEN"])
    num_features.extend(["DIAMETRE_LOG", "LENGTH_SQRT"])

    # On identifie les lignes où l'année est manquante
    df_missing = df[df['ANNEEPOSE'].isna()].copy()

    # Tu les prépares comme X_missing pour les passer au modèle
    X_missing = df_missing.drop(columns=['ANNEEPOSE'])
    # Faire le même préprocessing que sur X_train ici : dummies, fillna, etc.


    df_connu = df[df[target_column].notna()]
    X = df_connu.drop(columns = [target_column])


    """#--- Ajout v2---
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])
    X_missing[num_features] = scaler.transform(X_missing[num_features])"""

    y = df_connu[target_column]

    y = pd.to_datetime(y, format='%Y/%m/%d', errors='coerce')

    y = y.dt.year
    year_min = min(y)
    year_max = max(y)

    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    X = X.astype(float)
    y = y.astype(float)

    # On stocke les prédictions de tous les folds pour moyenne
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n🌿 Fold {fold+1}/5")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

        def objective_cv(trial):
            hidden_size = trial.suggest_int("hidden_size", 32, 128)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

            model = MCDropoutNet(input_dim=X_train.shape[1], hidden_size=hidden_size, dropout=dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_loss = float('inf')
            patience = 10
            counter = 0

            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_train_tensor)
                loss = F.l1_loss(y_pred, y_train_tensor)
                loss.backward()
                optimizer.step()

                # Early stopping tracking
                current_loss = loss.item()
                if current_loss < best_loss - 1e-4:  # tolérance pour éviter les arrêts trop rapides
                    best_loss = current_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"⏹️  Early stopping at epoch {epoch+1}, best training loss: {best_loss:.4f}")
                        break


            mean_pred, _ = predict_mc(model, X_val_tensor)
            val_mae = mean_absolute_error(mean_pred.cpu().detach().numpy(), y_val.to_numpy())
            return val_mae

        study_fold = optuna.create_study(direction="minimize")
        study_fold.optimize(objective_cv, n_trials=15)
        
        print("➡️ Meilleur MAE sur fold :", study_fold.best_value)
        cv_results.append(study_fold.best_value)






    # Conversion des colonnes booléennes en float
    X_train = X_train.astype(float)
    X_val = X_val.astype(float)
    X_missing = X_missing.astype(float)
    
    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_missing = torch.tensor(X_missing.values, dtype=torch.float32).to(device)

    scaler = StandardScaler()
    transformer = PowerTransformer()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_missing = scaler.transform(X_missing)
    X_train = transformer.fit_transform(X_train)
    X_val = transformer.transform(X_val)
    X_missing = transformer.transform(X_missing)
    
    
    return X_train, y_train, X_val, y_val, X_missing, df_missing, year_max, year_min