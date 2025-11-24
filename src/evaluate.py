import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent))

from model import BasicMethylationNet
from dataset import CpGMethylationDataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "per_chrom_npz"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_test_data(chromosome="chr19"):

    data_path = DATA_DIR / f"{chromosome}.npz"
    d = np.load(data_path, allow_pickle=True)
    
    dna, hist, meth = d["dna"], d["histone"], d["methyl"]
    
    # Use same split as training (last 20%)
    # same training split as in train.py
    N = len(meth)
    test_start = int(0.8 * N)
    
    return dna[test_start:], hist[test_start:], meth[test_start:]


def evaluate_model(model_path, chromosome="chr19"):

    
    print(f"Loading test data from {chromosome}...")
    dna_test, hist_test, meth_test = load_test_data(chromosome)
    
    print(f"Test samples: {len(meth_test):,}")
    
    # Create test loader
    test_loader = DataLoader(
        CpGMethylationDataset(dna_test, hist_test, meth_test),
        batch_size=64, shuffle=False
    )
    
    # load model
    print(f"\nLoading model from {model_path}...")
    model = BasicMethylationNet(dropout=0.3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # generate predictions
    print("Generating predictions...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for dna, hist, y in test_loader:
            dna, hist = dna.to(device), hist.to(device)
            pred = model(dna, hist)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    # compute metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r, _ = pearsonr(y_pred, y_true)
    spearman_rho, _ = spearmanr(y_pred, y_true)
    
    # print results
    print("\n" + "="*70)
    print(f"Test Set Performance ({chromosome})")
    print("="*70)
    print(f"MSE:        {mse:.6f}")
    print(f"MAE:        {mae:.6f}")
    print(f"Pearson r:  {pearson_r:.4f}")
    print(f"Spearman ρ: {spearman_rho:.4f}")
    print(f"R²:         {r2:.4f}")
    print("="*70)
    
    # save metrics
    metrics = {
        "chromosome": chromosome,
        "n_samples": len(y_true),
        "MSE": float(mse),
        "MAE": float(mae),
        "Pearson_r": float(pearson_r),
        "Spearman_rho": float(spearman_rho),
        "R2": float(r2)
    }
    
    metrics_path = RESULTS_DIR / f"{chromosome}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nSaved metrics → {metrics_path}")
    
    # save predictions
    preds_path = RESULTS_DIR / f"{chromosome}_predictions.npz"
    np.savez(preds_path, y_true=y_true, y_pred=y_pred)
    print(f"Saved predictions → {preds_path}")


if __name__ == "__main__":
    model_path = CHECKPOINT_DIR / "best_model.pth"
    evaluate_model(model_path, chromosome="chr19")