# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:03:54 2025

@author: mc
"""

import matplotlib.pyplot as plt

def plot_predictions(preds, stds):
    plt.figure(figsize=(12, 6))
    plt.errorbar(range(len(preds)), preds, yerr=stds, fmt='o', ecolor='red', alpha=0.6, label='Incertitude')
    plt.plot(range(len(preds)), preds, label='Prédictions', color='blue')
    plt.fill_between(range(len(preds)), preds - stds, preds + stds, color='gray', alpha=0.2, label='Intervalle de confiance')
    plt.xlabel("Index")
    plt.ylabel("Prédiction")
    plt.title("Prédictions avec incertitudes (MC Dropout)")
    plt.legend()
    plt.grid(True)
    plt.show()