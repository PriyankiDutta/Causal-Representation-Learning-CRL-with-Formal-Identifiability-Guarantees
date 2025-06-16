# run_experiments.py

import os
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

from models import SupervisedEncoder
from train_ivae import train_ivae
from train_vicreg import train_vicreg
from utils import save_model, save_metrics
from mig import compute_MIG


EXPERIMENTS = [
    {"name": "A1", "decoder": "linear",    "confounded": False, "z_dim": 6,  "noise": 0.0},
    {"name": "A2", "decoder": "nonlinear", "confounded": False, "z_dim": 6,  "noise": 0.0},
    {"name": "A3", "decoder": "nonlinear", "confounded": True,  "z_dim": 6,  "noise": 0.0},
    {"name": "A4", "decoder": "nonlinear", "confounded": False, "z_dim": 6,  "noise": 0.1},
    {"name": "A5", "decoder": "nonlinear", "confounded": False, "z_dim": 10, "noise": 0.0},
    {"name": "A6", "decoder": "nonlinear", "confounded": True,  "z_dim": 10, "noise": 0.1},
]

MODELS = ["supervised", "iVAE", "VICReg"]

OUTPUT_DIR = "results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def correlation_matching(z_true_np, z_pred_np, z_dim, save_prefix):
    corr = np.corrcoef(z_true_np.T, z_pred_np.T)[:z_dim, z_dim:]
    cost_matrix = -np.abs(corr)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_corrs = {f"z{i+1} ↔ ẑ{j+1}": float(corr[i, j]) for i, j in zip(row_ind, col_ind)}
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=[f"ẑ{i+1}" for i in range(z_dim)],
                yticklabels=[f"z{i+1}" for i in range(z_dim)])
    plt.title("Correlation: True z vs Predicted ẑ")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_corr_heatmap.png")
    plt.close()
    return matched_corrs


def load_data(config):
    name = config["name"]
    fname = f"data/synthetic_{name}.npz"
    data = np.load(fname)
    x, z = data["x"], data["z"]

    # Scale
    x = StandardScaler().fit_transform(x)
    z = StandardScaler().fit_transform(z)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)


def run_all():
    for config in EXPERIMENTS:
        x, z_true = load_data(config)
        z_dim = config["z_dim"]
        print(f"\n▶ Running config {config['name']} — z_dim={z_dim}, noise={config['noise']}, confounded={config['confounded']}")

        for model_name in MODELS:
            print(f"   🔧 Model: {model_name}")
            save_prefix = f"{OUTPUT_DIR}/{config['name']}_{model_name}"
            os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

            if model_name == "supervised":
                model = SupervisedEncoder(input_dim=x.shape[1], latent_dim=z_dim)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                criterion = torch.nn.MSELoss()

                for epoch in range(5000):
                    model.train()
                    optimizer.zero_grad()
                    z_pred = model(x)
                    loss = criterion(z_pred, z_true)
                    loss.backward()
                    optimizer.step()

                z_pred_np = z_pred.detach().numpy()
                save_model(model, save_prefix + "_model.pt")

            elif model_name == "iVAE":
                model, z_pred_np = train_ivae(x, z_true)
                save_model(model, save_prefix + "_encoder.pt")

            elif model_name == "VICReg":
                model, z_pred_np = train_vicreg(x, latent_dim=z_dim)
                save_model(model, save_prefix + "_model.pt")

            z_true_np = z_true.numpy()
            matched_corrs = correlation_matching(z_true_np, z_pred_np, z_dim, save_prefix)
            # ---- MIG computation ----
            mig_score, mi_matrix = compute_MIG(z_true_np, z_pred_np)
            matched_corrs["MIG"] = mig_score  # Add MIG to metrics

            # Save MI heatmap
            sns.heatmap(mi_matrix, annot=True, fmt=".2f", cmap="viridis",
                        xticklabels=[f"ẑ{i+1}" for i in range(z_pred_np.shape[1])],
                        yticklabels=[f"z{i+1}" for i in range(z_true_np.shape[1])])
            plt.title("Mutual Information: True z vs Predicted ẑ")
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_mi_heatmap.png")
            plt.close()

            save_metrics(matched_corrs, save_prefix + "_metrics.json")


if __name__ == "__main__":
    run_all()
