# generate_synthetic_data.py
import numpy as np

EXPERIMENTS = [
    {"name": "A1", "decoder": "linear",    "confounded": False, "z_dim": 6,  "noise": 0.0},
    {"name": "A2", "decoder": "nonlinear", "confounded": False, "z_dim": 6,  "noise": 0.0},
    {"name": "A3", "decoder": "nonlinear", "confounded": True,  "z_dim": 6,  "noise": 0.0},
    {"name": "A4", "decoder": "nonlinear", "confounded": False, "z_dim": 6,  "noise": 0.1},
    {"name": "A5", "decoder": "nonlinear", "confounded": False, "z_dim": 10, "noise": 0.0},
    {"name": "A6", "decoder": "nonlinear", "confounded": True,  "z_dim": 10, "noise": 0.1},
]

def generate_data(z_dim, decoder="nonlinear", confounded=False, noise=0.0, n_samples=10000):
    z = np.random.randn(n_samples, z_dim)
    
    if confounded:
        u = np.random.randn(n_samples, z_dim)
        z += 0.5 * u  # Simple confounding

    if decoder == "linear":
        W = np.random.randn(z_dim, 16)
        x = z @ W
    else:
        x = np.tanh(z @ np.random.randn(z_dim, 32)) @ np.random.randn(32, 16)

    x += noise * np.random.randn(*x.shape)
    return x, z

for cfg in EXPERIMENTS:
    x, z = generate_data(cfg['z_dim'], cfg['decoder'], cfg['confounded'], cfg['noise'])
    np.savez(f"data/synthetic_{cfg['name']}.npz", x=x, z=z)
    print(f"✅ Saved data/synthetic_{cfg['name']}.npz with z_dim={cfg['z_dim']}")
