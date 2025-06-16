# Causal-Representation-Learning-CRL-with-Formal-Identifiability-Guarantees


This project investigates **identifiability of latent representations** using synthetic structural causal models (SCMs). It compares different models — including **Supervised Encoders**, **iVAE**, and **VICReg** — on their ability to recover true latent variables from observed data.

Key evaluation tools include **correlation analysis**, **Hungarian matching**, **heatmap visualization**, and **MIG (Mutual Information Gap)**.

---

## 📦 Features

* ✅ **Synthetic Data Generator** with configurable latent dimensions, noise, nonlinear decoders, and confounding.
* 🧠 **Model Training Pipelines** for:

  * Supervised Encoder
  * iVAE (Identifiable VAE)
  * VICReg (Invariant Contrastive Learning)
* 📊 **Identifiability Metrics**:

  * Correlation heatmaps
  * Hungarian matching
  * Mutual Information Gap (MIG)
* 📁 Modular structure to easily extend with new models or data configs.
* 💾 Model saving and experiment logging.

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/identifiability-crl.git
cd identifiability-crl
pip install -r requirements.txt
```

Make sure you have Python 3.8+ and PyTorch installed.

---

## 🚀 Usage

Run experiments with all 3 models on various synthetic configurations:

```bash
python run_experiments.py
```

You can customize settings like latent dimensions, nonlinearity, and confounding directly in `run_experiments.py`.

---

## 📁 Project Structure

```
.
├── run_experiments.py       # Main experiment runner
├── models/                  # Supervised, iVAE, VICReg models
├── train_ivae.py            # iVAE training script
├── train_vicreg.py          # VICReg training script
├── utils/
│   ├── metrics.py           # Identifiability metrics
│   ├── mig.py               # Mutual Information Gap (optional)
│   ├── visualization.py     # Heatmaps and correlation plots
│   └── save_model.py        # Model saving utilities
├── synthetic_data.py        # SCM-based synthetic data generator
└── requirements.txt
```

---

## 📈 Example Output

* 🔗 Correlation heatmaps between true latents and learned representations
* ✅ Matching accuracy via the Hungarian algorithm
* 📊 MIG scores

---


---

## 👩‍💻 Author

**Priyanki Dutta**
For questions or collaborations, contact via GitHub or email.


