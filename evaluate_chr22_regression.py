import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json



print("Loading predictions...")
preds = np.load("chr22_preds.npy").reshape(-1)
coords_test = np.load("chr22_coords_test.npy", allow_pickle=True)

print("Loading ground-truth from chr22.npz...")
data = np.load("per_chrom_npz/chr22.npz", allow_pickle=True)


keys = data.files
print("Available keys:", keys)


coord_key = next(k for k in keys if "coord" in k.lower())
meth_key  = next(k for k in keys if "meth"  in k.lower())

coords_all = data[coord_key]
meth_all   = data[meth_key].reshape(-1)

print(f"coords_all shape = {coords_all.shape}")
print(f"meth_all shape   = {meth_all.shape}")


# ----------------------------
# Match test coordinates
# ----------------------------
print("Matching coordinates...")

coord_to_index = {tuple(c): i for i, c in enumerate(coords_all)}

test_indices = []
for coord in coords_test:
    tpl = tuple(coord)
    if tpl not in coord_to_index:
        raise ValueError(f"Coordinate {tpl} missing in chr22.npz")
    test_indices.append(coord_to_index[tpl])

test_indices = np.array(test_indices)

y_true = meth_all[test_indices]
y_pred = preds

print(f"Matched {len(y_true)} test samples.")


# ----------------------------
# Compute regression metrics
# ----------------------------
mse  = mean_squared_error(y_true, y_pred)
mae  = mean_absolute_error(y_true, y_pred)
r    = pearsonr(y_true, y_pred)[0]
rho  = spearmanr(y_true, y_pred)[0]
r2   = r2_score(y_true, y_pred)

print("\n=== chr22 Regression Performance ===")
print(f"MSE       = {mse:.6f}")
print(f"MAE       = {mae:.6f}")
print(f"Pearson r = {r:.4f}")
print(f"Spearman ρ = {rho:.4f}")
print(f"R²        = {r2:.4f}")

# save results
metrics = {
    "MSE": float(mse),
    "MAE": float(mae),
    "Pearson_r": float(r),
    "Spearman_rho": float(rho),
    "R2": float(r2)
}


with open("chr22_regression_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nSaved → chr22_regression_metrics.json")
