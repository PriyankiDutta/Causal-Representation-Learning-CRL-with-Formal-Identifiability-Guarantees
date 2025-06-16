# utils/mig.py
import numpy as np
from sklearn.metrics import mutual_info_score

def discretize(arr, bins=20):
    binned = np.digitize(arr, np.histogram(arr, bins=bins)[1][:-1])
    return binned

def compute_mutual_info_matrix(z_true, z_pred, bins=20):
    z_true_d = np.array([discretize(z_true[:, i], bins) for i in range(z_true.shape[1])])
    z_pred_d = np.array([discretize(z_pred[:, j], bins) for j in range(z_pred.shape[1])])
    
    mi_matrix = np.zeros((z_true.shape[1], z_pred.shape[1]))

    for i in range(z_true.shape[1]):
        for j in range(z_pred.shape[1]):
            mi_matrix[i, j] = mutual_info_score(z_true_d[i], z_pred_d[j])
    
    return mi_matrix

def compute_MIG(z_true, z_pred, bins=20):
    mi_matrix = compute_mutual_info_matrix(z_true, z_pred, bins)
    entropies = [mutual_info_score(discretize(z_true[:, i], bins), discretize(z_true[:, i], bins)) for i in range(z_true.shape[1])]

    migs = []
    for i in range(z_true.shape[1]):
        sorted_mis = np.sort(mi_matrix[i])[::-1]
        if entropies[i] > 0:
            mig = (sorted_mis[0] - sorted_mis[1]) / entropies[i]
            migs.append(mig)
    
    return float(np.mean(migs)), mi_matrix
