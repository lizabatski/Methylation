import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# add src to path
sys.path.append(str(Path(__file__).parent))

from model import BasicMethylationNet
from dataset import CpGMethylationDataset
from tqdm import tqdm
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "per_chrom_npz"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def load_data(chromosome="chr19"):
    data_path = DATA_DIR / f"{chromosome}.npz"
    print(f"Loading {data_path}...")
    
    d = np.load(data_path, allow_pickle=True)
    # Skip corrupted histone_names, use hardcoded order
    return d["dna"], d["histone"], d["methyl"]


def simple_split(dna, hist, meth, train_frac=0.6, val_frac=0.2):
    """Split data into train/val/test"""
    N = len(meth)
    i1 = int(train_frac * N)
    i2 = int((train_frac + val_frac) * N)
    
    return (
        dna[:i1], hist[:i1], meth[:i1],
        dna[i1:i2], hist[i1:i2], meth[i1:i2],
        dna[i2:], hist[i2:], meth[i2:]
    )


def train_epoch(model, loader, optimizer, loss_fn):
    """Single training epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for dna, hist, y in pbar:
        dna, hist, y = dna.to(device), hist.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(dna, hist)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


def validate(model, loader, loss_fn):
    """Validation with metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for dna, hist, y in tqdm(loader, desc="Validation"):
            dna, hist, y = dna.to(device), hist.to(device), y.to(device)
            pred = model(dna, hist)
            loss = loss_fn(pred, y)
            
            total_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    pearson_r, _ = pearsonr(all_preds, all_targets)
    spearman_rho, _ = spearmanr(all_preds, all_targets)
    
    return {
        'loss': total_loss / len(loader),
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_r,
        'spearman_rho': spearman_rho,
        'pred_mean': all_preds.mean(),
        'pred_std': all_preds.std(),
        'pred_range': (all_preds.min(), all_preds.max()),
    }


def train(model, train_loader, val_loader, epochs=50, lr=1e-4):
    """Full training loop"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    loss_fn = torch.nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7
    
    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_metrics = validate(model, val_loader, loss_fn)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val MSE:    {val_metrics['mse']:.4f}")
        print(f"  Val R²:     {val_metrics['r2']:.4f}")
        print(f"  Pearson r:  {val_metrics['pearson_r']:.4f}")
        
        scheduler.step(val_metrics['loss'])
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / 'best_model.pth')
            print(f"  ✓ Saved best model")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping!")
                break
    
    print("\n" + "="*70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print("="*70)


if __name__ == "__main__":
    dna, hist, meth = load_data(chromosome="chr19")
    
    print(f"\nDataset info:")
    print(f"  Total samples: {len(meth):,}")
    print(f"  DNA shape: {dna.shape}")
    print(f"  Histone shape: {hist.shape}")
    print(f"  Methylation range: [{meth.min():.3f}, {meth.max():.3f}]")
    
    dna_train, hist_train, meth_train, \
    dna_val, hist_val, meth_val, \
    dna_test, hist_test, meth_test = simple_split(dna, hist, meth)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(meth_train):,}")
    print(f"  Val:   {len(meth_val):,}")
    print(f"  Test:  {len(meth_test):,}")
    
    # dataloaders
    train_loader = DataLoader(
        CpGMethylationDataset(dna_train, hist_train, meth_train),
        batch_size=64, shuffle=True, num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        CpGMethylationDataset(dna_val, hist_val, meth_val),
        batch_size=64, shuffle=False, num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # initialize model
    model = BasicMethylationNet(dropout=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    
    # train
    print("\nStarting training...\n")
    train(model, train_loader, val_loader, epochs=50, lr=1e-4)