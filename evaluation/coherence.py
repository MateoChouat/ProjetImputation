# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:56:17 2025

@author: mc
"""
"""
def check_coherence(row, predicted_year, year_max, year_min):
    errors = []
    if predicted_year < year_min or predicted_year > year_max:
        errors.append("Année hors des bornes")
    if row.get('MATERIAU_Fonte grise', 0) and predicted_year > 1960:
        errors.append("Fonte grise utilisée après 1960")
    if row.get('DIAMETRE', 0) < 2000 and predicted_year < 1900:
        errors.append("Gros diamètre trop ancien")
    return errors

def apply_coherence_check(df_missing, preds, stds, year_max, year_min):
    results = []
    for i, (mean,std) in enumerate(zip(preds, stds)):
        row = df_missing.iloc[i]
        errors = check_coherence(row, mean, year_max, year_min)
        results.append({
            "Index": i,
            "Prédiction": round(mean, 1),
            "Incertitude": round(std, 1),
            "Incohérences": errors
        })
    return results
"""

import pandas as pd

def check_coherence(row, predicted_year, year_max, year_min):
    """
    Vérifie si l'année prédite et les données de la ligne respectent les règles définies.

    Parameters:
    - row (pd.Series): Une ligne unique du DataFrame contenant les colonnes nécessaires.
    - predicted_year (float): L'année prédite à évaluer.
    - year_max (int): Année maximale autorisée.
    - year_min (int): Année minimale autorisée.

    Returns:
    - errors (list): Liste des incohérences détectées pour cette ligne.
    """
    errors = []
    if predicted_year < year_min or predicted_year > year_max:
        errors.append("Année hors des bornes")
    if row.get('MATERIAU_Fonte grise', 0) and predicted_year > 1960:
        errors.append("Fonte grise utilisée après 1960")
    if row.get('DIAMETRE', 0) < 2000 and predicted_year < 1900:
        errors.append("Gros diamètre trop ancien")
    return errors


def apply_coherence_check(df_missing, preds, stds, year_max, year_min):
    """
    Applique les vérifications de cohérence à toutes les lignes du DataFrame.

    Parameters:
    - df_missing (pd.DataFrame): DataFrame contenant les données manquantes à vérifier.
    - preds (list ou np.array): Valeurs moyennes prédites pour chaque ligne.
    - stds (list ou np.array): Écarts-types pour chaque prédiction.
    - year_max (int): Année maximale autorisée.
    - year_min (int): Année minimale autorisée.

    Returns:
    - results_df (pd.DataFrame): DataFrame contenant les prédictions, incertitudes et incohérences.
    """
    results = []
    for i, (mean, std) in enumerate(zip(preds, stds)):
        row = df_missing.iloc[i]
        errors = check_coherence(row, mean, year_max, year_min)
        results.append({
            "Index": i,
            "Prédiction": round(mean, 1),
            "Incertitude": round(std, 1),
            "Incohérences": "; ".join(errors) if errors else "Aucune"
        })

    # Convertir la liste des résultats en DataFrame pour un affichage formaté
    results_df = pd.DataFrame(results)
    print("\n=== Résultats des vérifications de cohérence ===\n")
    print(results_df.to_string(index=False))
    return results_df


