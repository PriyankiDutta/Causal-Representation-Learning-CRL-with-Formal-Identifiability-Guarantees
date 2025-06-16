# train_ivae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

class iVAE_Encoder(nn.Module):
    def __init__(self, x_dim, u_dim, z_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(x_dim + u_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mu = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)

    def forward(self, x, u):
        h = self.fc(torch.cat([x, u], dim=-1))
        return self.mu(h), self.logvar(h)

class iVAE_Prior(nn.Module):
    def __init__(self, u_dim, z_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(u_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, z_dim)
        self.logvar = nn.Linear(64, z_dim)

    def forward(self, u):
        h = self.fc(u)
        return self.mu(h), self.logvar(h)

class iVAE_Decoder(nn.Module):
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, x_dim)
        )

    def forward(self, z):
        return self.model(z)

def sample_z(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_divergence(q_mu, q_logvar, p_mu, p_logvar):
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    return 0.5 * ((p_logvar - q_logvar + (q_var + (q_mu - p_mu)**2) / p_var) - 1).sum(dim=1).mean()

def train_ivae(x, z_true, K=10, epochs=5000, lr=1e-3, beta=5.0):
    z_np = z_true.numpy()
    u = KMeans(n_clusters=K, random_state=0).fit_predict(z_np)
    u_onehot = torch.tensor(np.eye(K)[u], dtype=torch.float32)

    encoder = iVAE_Encoder(x.shape[1], K, z_true.shape[1])
    prior = iVAE_Prior(K, z_true.shape[1])
    decoder = iVAE_Decoder(z_true.shape[1], x.shape[1])

    params = list(encoder.parameters()) + list(prior.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(epochs):
        encoder.train(); decoder.train(); prior.train()
        optimizer.zero_grad()

        q_mu, q_logvar = encoder(x, u_onehot)
        z_sample = sample_z(q_mu, q_logvar)
        x_recon = decoder(z_sample)

        recon_loss = F.mse_loss(x_recon, x)
        p_mu, p_logvar = prior(u_onehot)
        kl = kl_divergence(q_mu, q_logvar, p_mu, p_logvar)

        loss = recon_loss + beta * kl
        loss.backward()
        optimizer.step()

    return encoder, q_mu.detach().numpy()
