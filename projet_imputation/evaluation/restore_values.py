# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 08:50:42 2025

@author: mc
"""

from sklearn.preprocessing import StandardScaler

def retransform(X_train, X_val, X_missing): 
    scaler = StandardScaler()
    X_train_original = scaler.inverse_transform(X_train)
    X_val_original = scaler.inverse_transform(X_val)
    X_missing_original = scaler.invers_transform(X_missing)
    return X_train_original, X_val_original, X_missing_original