import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from model import BasicMethylationNet
from dataset import CpGMethylationDataset
from torch.utils.data import DataLoader

print("="*70)
print("TESTING METHYLATION MODEL")
print("="*70)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n✓ Device: {device}")
if device == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test model creation
print("\n" + "-"*70)
print("Testing model creation...")
print("-"*70)

model = BasicMethylationNet(
    dna_filters=[64, 128, 128],  # Smaller for testing
    dna_kernel_sizes=[15, 11, 7],
    histone_filters=16,
    dropout=0.3,
    fc_hidden_dims=[64, 32]
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Model created successfully")
print(f"  Total parameters: {n_params:,}")

# Test forward pass with dummy data
print("\n" + "-"*70)
print("Testing forward pass with dummy data...")
print("-"*70)

batch_size = 4
dummy_dna = torch.randn(batch_size, 500, 4).to(device)
dummy_histone = torch.randn(batch_size, 500, 4).to(device)

try:
    output = model(dummy_dna, dummy_histone)
    print(f"✓ Forward pass successful")
    print(f"  Input DNA shape: {dummy_dna.shape}")
    print(f"  Input histone shape: {dummy_histone.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load real data
print("\n" + "-"*70)
print("Testing with real data...")
print("-"*70)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "per_chrom_npz"

# Check if directory exists
if not DATA_DIR.exists():
    print(f"✗ Data directory not found: {DATA_DIR}")
    print(f"  Current directory: {Path.cwd()}")
    print(f"  Looking for: {DATA_DIR.absolute()}")
    sys.exit(1)

try:
    data_path = DATA_DIR / "chr19.npz"
    if not data_path.exists():
        # Try without subdirectory
        data_path = Path("per_chrom_npz/chr19.npz")
        if not data_path.exists():
            print(f"✗ chr19.npz not found in:")
            print(f"  {DATA_DIR / 'chr19.npz'}")
            print(f"  {data_path}")
            sys.exit(1)
    
    print(f"Loading: {data_path}")
    
    d = np.load(data_path, allow_pickle=True)
    dna = d["dna"]
    histone = d["histone"]
    methyl = d["methyl"]
    
    print(f"✓ Data loaded successfully")
    print(f"  Samples: {len(methyl):,}")
    print(f"  DNA shape: {dna.shape}")
    print(f"  Histone shape: {histone.shape}")
    print(f"  Methylation range: [{methyl.min():.3f}, {methyl.max():.3f}]")
    print(f"  Methylation mean: {methyl.mean():.3f} ± {methyl.std():.3f}")
    
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test dataset and dataloader
print("\n" + "-"*70)
print("Testing dataset and dataloader...")
print("-"*70)

try:
    # Use first 100 samples for quick test
    test_dataset = CpGMethylationDataset(
        dna[:100], 
        histone[:100], 
        methyl[:100]
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0  # Use 0 for testing
    )
    
    print(f"✓ Dataset created: {len(test_dataset)} samples")
    print(f"✓ DataLoader created: {len(test_loader)} batches")
    
except Exception as e:
    print(f"✗ Dataset/DataLoader failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test training step
print("\n" + "-"*70)
print("Testing single training step...")
print("-"*70)

try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    
    # Get one batch
    dna_batch, hist_batch, y_batch = next(iter(test_loader))
    dna_batch = dna_batch.to(device)
    hist_batch = hist_batch.to(device)
    y_batch = y_batch.to(device)
    
    # Forward pass
    pred = model(dna_batch, hist_batch)
    loss = loss_fn(pred, y_batch)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step successful")
    print(f"  Batch size: {len(y_batch)}")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Predictions range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
    print(f"  Targets range: [{y_batch.min().item():.3f}, {y_batch.max().item():.3f}]")
    
except Exception as e:
    print(f"✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test validation step
print("\n" + "-"*70)
print("Testing validation step...")
print("-"*70)

try:
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for dna_batch, hist_batch, y_batch in test_loader:
            dna_batch = dna_batch.to(device)
            hist_batch = hist_batch.to(device)
            
            pred = model(dna_batch, hist_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error
    
    mse = mean_squared_error(all_targets, all_preds)
    r, _ = pearsonr(all_preds, all_targets)
    
    print(f"✓ Validation successful")
    print(f"  Samples evaluated: {len(all_preds)}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Pearson r: {r:.4f}")
    
except Exception as e:
    print(f"✗ Validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED! ✓")
print("="*70)
print("\nYour model is ready for training!")