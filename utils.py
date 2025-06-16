# utils.py

import os
import torch
import json

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_metrics(metrics_dict, path):
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
