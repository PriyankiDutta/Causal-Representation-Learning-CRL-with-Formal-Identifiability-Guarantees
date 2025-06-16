# train_vicreg.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VICRegEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

def invariance_loss(z1, z2):
    return F.mse_loss(z1, z2)

def variance_loss(z):
    std = torch.std(z, dim=0)
    return torch.mean(F.relu(1 - std))

def covariance_loss(z):
    z = z - z.mean(dim=0)
    N, D = z.size()
    cov = (z.T @ z) / (N - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / D

def train_vicreg(x, latent_dim, epochs=5000, lr=1e-3, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0):
    model = VICRegEncoder(x.shape[1], latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z1 = model(x)
        z2 = model(x)  # VICReg uses same input twice (can add noise for real setup)

        sim_loss = invariance_loss(z1, z2)
        var_loss = variance_loss(z1) + variance_loss(z2)
        cov_loss = covariance_loss(z1) + covariance_loss(z2)

        loss = sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss
        loss.backward()
        optimizer.step()

    return model, z1.detach().numpy()
