# My Utility : 
import numpy as np

# Normalización de los datos
def normalize_data(X):
    normalized_X = X / np.sqrt(len(X.columns) - 1)
    return normalized_X

# SVD de los datos
def svd_data(X, Y, param):
    # Restar la media de las columnas
    column_means = X.mean()
    X_centered = X - column_means
    
    # Normalizar los datos
    X_normalized = normalize_data(X_centered)
    
    # Realizar SVD
    U, S, V = np.linalg.svd(X_normalized)
    
    # Seleccionar las primeras 'param[2]' columnas de Vt
    V = V[:, :int(param[2])]
    
    return V