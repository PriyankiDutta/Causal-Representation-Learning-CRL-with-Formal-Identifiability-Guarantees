# evaluation.py

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def correlation_matrix(z_true, z_pred, name=""):
    corr = np.corrcoef(z_true.T, z_pred.T)[:z_true.shape[1], z_true.shape[1]:]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Correlation: True z vs {name}")
    plt.show()
    return corr

def hungarian_match(corr):
    cost_matrix = -np.abs(corr)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = [(i, j, corr[i, j]) for i, j in zip(row_ind, col_ind)]
    return matches

def print_matches(matches, name=""):
    print(f"\nMatching for {name}:")
    for i, j, c in matches:
        print(f"z{i+1} ↔ ẑ{j+1}, corr = {c:.2f}")

def compute_mig_placeholder(z_true, z_pred):
    return 0.0  # To be implemented
