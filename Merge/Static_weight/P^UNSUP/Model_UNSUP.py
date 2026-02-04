import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns  # for heatmap

# ========= Global settings =========
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14

sns.set_style("whitegrid")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# =========================
# 1. Data loading and preprocessing
# =========================
input_path = r"C:\Users\沐阳\Desktop\城市综合指数_pro\City_Data_Standardized_Results.xlsx"
df = pd.read_excel(input_path)

# Select features (columns from index 8 onward) and drop entirely empty columns
features = df.iloc[:, 8:].copy()
features = features.dropna(axis=1, how='all')

# Encode non-numeric columns and impute remaining missing values by median
for col in features.columns:
    if not np.issubdtype(features[col].dtype, np.number):
        features[col] = features[col].astype("category").cat.codes
    features[col] = features[col].fillna(features[col].median())

# Scale to [0,1] for AutoEncoder training
scaler = MinMaxScaler()
X = scaler.fit_transform(features.values.astype(np.float32))
X_tensor = torch.tensor(X, dtype=torch.float32)
input_dim = X_tensor.shape[1]

# Train / validation split
X_train, X_val = train_test_split(X, test_size=0.3, random_state=RANDOM_STATE)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

# =========================
# 2. Define AutoEncoder model
# =========================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(AutoEncoder, self).__init__()
        # Encoder: input -> 128 -> 64 -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Decoder: latent -> 64 -> 128 -> input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # outputs in [0,1] consistent with MinMaxScaler
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# =========================
# 3. Training configuration
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(input_dim=input_dim, latent_dim=16).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

batch_size = 256
train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor), batch_size=batch_size, shuffle=False)

num_epochs = 500
patience = 20
best_val_loss = float("inf")
pat_count = 0
best_state = None
train_loss_history, val_loss_history = [], []

# =========================
# 4. Training loop with early stopping
# =========================
for epoch in range(num_epochs):
    model.train()
    train_epoch_loss = 0.0
    for (xb,) in train_loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, xb)
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item() * xb.size(0)
    train_epoch_loss /= len(X_train_tensor)

    model.eval()
    val_epoch_loss = 0.0
    with torch.no_grad():
        for (xb,) in val_loader:
            xb = xb.to(device)
            pred = model(xb)
            val_loss = criterion(pred, xb)
            val_epoch_loss += val_loss.item() * xb.size(0)
    val_epoch_loss /= len(X_val_tensor)

    train_loss_history.append(train_epoch_loss)
    val_loss_history.append(val_epoch_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_epoch_loss:.6f} | Val Loss: {val_epoch_loss:.6f}")

    # Early stopping logic (monitor validation loss)
    if val_epoch_loss < best_val_loss - 1e-6:
        best_val_loss = val_epoch_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        pat_count = 0
    else:
        pat_count += 1
        if pat_count >= patience:
            print(f"Early stopping at epoch {epoch+1}, best_val_loss={best_val_loss:.6f}")
            break

# Load best model state
if best_state is not None:
    model.load_state_dict(best_state)
model.eval()

# =========================
# 5. Training vs validation loss plot
# =========================
output_dir = r"C:\Users\沐阳\Desktop\模型3.0输出结果\静态_无监督"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(8,5))
plt.plot(train_loss_history, label="Train Loss", color="#E45756", lw=2)
plt.plot(val_loss_history, label="Validation Loss", color="#4C78A8", lw=2)
plt.xlabel("Epoch", fontweight="bold")
plt.ylabel("MSE Loss", fontweight="bold")
plt.title("AutoEncoder Training vs Validation Loss Curve", fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

loss_fig_path = os.path.join(output_dir, "autoencoder_train_val_loss_curve.png")
plt.savefig(loss_fig_path, dpi=200)
plt.show()

# =========================
# 6. Reconstruction error analysis
# =========================
X_full_tensor = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    reconstructed = model(X_full_tensor.to(device)).cpu()

# Per-feature and per-sample mean squared reconstruction error
recon_error_per_feature = ((X_full_tensor - reconstructed) ** 2).mean(dim=0).numpy()
recon_error_per_sample = ((X_full_tensor - reconstructed) ** 2).mean(dim=1).numpy()
global_mse = ((X_full_tensor - reconstructed) ** 2).mean().item()

print(f"Global mean reconstruction error: {global_mse:.6f}")

# =========================
# 7. Compute feature weights from reconstruction error
#    weight_i = normalized (1 / error_i)
# =========================
inv_error = 1.0 / (recon_error_per_feature + 1e-8)
weights = inv_error / np.sum(inv_error)

weight_df = pd.DataFrame({
    'Feature': features.columns,
    'Weight': weights,
    'ReconstructionError': recon_error_per_feature
}).sort_values(by='Weight', ascending=False).reset_index(drop=True)

excel_path = os.path.join(output_dir, "feature_weights_autoencoder_with_val.xlsx")
weight_df.to_excel(excel_path, index=False)
print(f"Feature weights exported to: {excel_path}")

# =========================
# 8. Visualization: Top-K weights and distribution
# =========================
topk = 20
top_df = weight_df.head(topk)

plt.figure(figsize=(10, 6))
bars = plt.barh(top_df["Feature"][::-1], top_df["Weight"][::-1], color="#4C78A8")
plt.xlabel("Weight")
plt.title(f"Top-{topk} Feature Weights by AutoEncoder")
for i, v in enumerate(top_df["Weight"][::-1]):
    plt.text(v, i, f"{v:.4f}", va="center", ha="left", fontsize=9)
plt.tight_layout()

png_path = os.path.join(output_dir, "feature_weights_autoencoder_top20_with_val.png")
plt.savefig(png_path, dpi=200)
plt.show()

plt.figure(figsize=(8,5))
plt.hist(weights, bins=30, color='#4C78A8', edgecolor='black', alpha=0.7)
plt.xlabel("Weight Value", fontweight="bold")
plt.ylabel("Count", fontweight="bold")
plt.title("Feature Weight Distribution", fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

hist_path = os.path.join(output_dir, "feature_weight_distribution_with_val.png")
plt.savefig(hist_path, dpi=200)
plt.show()

# =========================
# 9. Reconstruction error visualizations
# =========================

# (1) Heatmap of reconstruction error for samples x features (subset of samples)
subset_size = min(100, len(X_full_tensor))
heatmap_data = ((X_full_tensor - reconstructed) ** 2).numpy()[:subset_size, :]

plt.figure(figsize=(12, 6))
sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    xticklabels=features.columns,
    yticklabels=False,
    cbar_kws={'label': 'Reconstruction Error'}
)
plt.title(f"Reconstruction Error Heatmap (First {subset_size} Samples × {input_dim} Features)")
plt.xlabel("Feature")
plt.ylabel("Sample Index")
plt.tight_layout()

heatmap_path = os.path.join(output_dir, "reconstruction_error_heatmap.png")
plt.savefig(heatmap_path, dpi=200)
plt.show()
print(f"Reconstruction error heatmap saved to: {heatmap_path}")

# (2) Top-20 samples by mean reconstruction error
sample_error_df = pd.DataFrame({
    'SampleIndex': np.arange(len(recon_error_per_sample)),
    'ReconstructionError': recon_error_per_sample
})
top_samples = sample_error_df.sort_values(by='ReconstructionError', ascending=False).head(20)

plt.figure(figsize=(10, 5))
plt.bar(top_samples['SampleIndex'].astype(str), top_samples['ReconstructionError'], color='#E45756')
plt.xticks(rotation=45)
plt.xlabel("Sample Index (Top 20)")
plt.ylabel("Reconstruction Error")
plt.title("Top-20 Samples with Highest Reconstruction Error")
plt.tight_layout()

sample_err_path = os.path.join(output_dir, "top20_samples_reconstruction_error.png")
plt.savefig(sample_err_path, dpi=200)
plt.show()
print(f"Top-20 sample reconstruction error plot saved to: {sample_err_path}")

# =========================
# Summary message
# =========================
print("\nAll steps completed:")
print("1) Train/validation loss curves (monitoring for overfitting)")
print("2) Feature weights and distribution exported")
print("3) Reconstruction error heatmap (feature × sample)")
print("4) Top-20 samples by reconstruction error")