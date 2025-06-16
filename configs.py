# configs.py

experiment_grid = [
    {"name": "A1", "decoder": "linear", "confounded": False, "latent_dim": 6, "noise_std": 0.0},
    {"name": "A2", "decoder": "nonlinear", "confounded": False, "latent_dim": 6, "noise_std": 0.0},
    {"name": "A3", "decoder": "nonlinear", "confounded": True, "latent_dim": 6, "noise_std": 0.0},
    {"name": "A4", "decoder": "nonlinear", "confounded": False, "latent_dim": 6, "noise_std": 0.1},
    {"name": "A5", "decoder": "nonlinear", "confounded": False, "latent_dim": 10, "noise_std": 0.0},
    {"name": "A6", "decoder": "nonlinear", "confounded": True, "latent_dim": 10, "noise_std": 0.1},
]
