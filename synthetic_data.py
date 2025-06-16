# synthetic_data.py

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def generate_synthetic_data(decoder_type, confounded, latent_dim, noise_std):
    """
    Replace this stub with your own data generation logic.
    Must return torch tensors: x, z (standardized)
    """
    # Fake data for now — REPLACE this with your real generator
    N, x_dim = 10000, 16
    z = np.random.randn(N, latent_dim)
    x = z @ np.random.randn(latent_dim, x_dim) + noise_std * np.random.randn(N, x_dim)

    # Apply confounding (optional)
    if confounded:
        z += 0.5 * np.sin(z)

    # Standardize
    x = StandardScaler().fit_transform(x)
    z = StandardScaler().fit_transform(z)

    x = torch.tensor(x, dtype=torch.float32)
    z = torch.tensor(z, dtype=torch.float32)

    return x, z
