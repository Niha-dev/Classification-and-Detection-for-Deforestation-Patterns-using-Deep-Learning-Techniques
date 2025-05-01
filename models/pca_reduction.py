import os
import numpy as np
import joblib
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ✅ Paths
feature_path = "C:/Users/nniha/flask_env/Scripts/project/models/features.npy"
pca_feature_path = "C:/Users/nniha/flask_env/Scripts/project/models/X_pca.npy"
pca_model_path = "C:/Users/nniha/flask_env/Scripts/project/models/pca_mod.pkl"
pca_variance_path = "C:/Users/nniha/flask_env/Scripts/project/models/pca_variance.npy"

# ✅ Load features safely
if not os.path.exists(feature_path):
    raise FileNotFoundError("❌ Error: Features file not found!")

X = np.load(feature_path)

# ✅ Handle empty dataset
if X.size == 0 or X.shape[0] == 0:
    raise ValueError("❌ Error: No features found in the dataset!")

if X.shape[0] < 2:
    raise ValueError("❌ Error: At least 2 samples are required for PCA!")

# ✅ Reshape if features are in (samples, 1, 2048) format
if len(X.shape) == 3 and X.shape[1] == 1:
    X = X.reshape(X.shape[0], X.shape[2])  # Reshape to (samples, 2048)
elif len(X.shape) != 2:
    raise ValueError(f"❌ Error: Unexpected feature shape: {X.shape}. Expected (samples, features).")

# ✅ Handle `NaN` or corrupted values
if not np.isfinite(X).all():
    raise ValueError("❌ Error: Dataset contains NaN or infinite values! Check preprocessing.")

# ✅ Standardize features before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Find optimal `n_components` to preserve 95% variance
pca_full = PCA()
pca_full.fit(X_scaled)  # Fit PCA to compute explained variance
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)

# ✅ Select `n_components` where variance ≥ 95%
n_components = np.argmax(explained_variance >= 0.95) + 1
n_components = max(2, min(n_components or min(64, X.shape[1]), 64))

# ✅ Apply PCA with optimal components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# ✅ Prevent accidental overwriting of PCA files
# ✅ Prevent accidental overwriting of PCA files
for path in [pca_feature_path, pca_model_path, pca_variance_path]:
    backup_path = path + ".backup"
    if os.path.exists(backup_path):
        os.remove(backup_path)  # Delete old backup
    if os.path.exists(path):
        os.rename(path, backup_path)  # Rename safely



print("🔄 Previous PCA files backed up (if existed). Saving new files...")

# ✅ Save reduced features, PCA model, and explained variance
try:
    np.save(pca_feature_path, X_pca)
    joblib.dump(pca, pca_model_path)
    np.save(pca_variance_path, pca.explained_variance_ratio_)
except Exception as e:
    print(f"❌ Error saving PCA files: {e}")
    exit(1)
finally:
    print("🔄 File save operation completed. Check logs if any issues.")

# ✅ Print Summary
print(f"✅ PCA Features saved at: {pca_feature_path}")
print(f"💾 PCA Model saved at: {pca_model_path}")
print(f"📊 Original Features Shape: {X.shape}")
print(f"📉 Reduced PCA Features Shape: {X_pca.shape}")
print(f"🔎 Selected Components: {n_components}")
print(f"📈 Explained Variance Preserved: {explained_variance[n_components-1]:.4f}")
