# train_supervised.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import pairwise_distances
import numpy as np

class SupervisedEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.model(x)

def train_supervised(x, z, epochs=5000, lr=1e-3):
    model = SupervisedEncoder(x.shape[1], z.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z_pred = model(x)
        loss = criterion(z_pred, z)
        loss.backward()
        optimizer.step()
    return model, z_pred.detach().numpy()
