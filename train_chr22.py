# I think imports were taking a very long time
import time

t0 = time.time()
print("[0] Starting imports...")

print("[1] Importing numpy")
import numpy as np
print(f"    numpy imported in {time.time() - t0:.2f}s")

t1 = time.time()
print("[2] Importing torch")
import torch
print(f"    torch imported in {time.time() - t1:.2f}s")

t2 = time.time()
print("[3] Importing DataLoader")
from torch.utils.data import DataLoader
print(f"    DataLoader imported in {time.time() - t2:.2f}s")

t3 = time.time()
print("[4] Importing BasicMethylationNet")
from model import BasicMethylationNet
print(f"    BasicMethylationNet imported in {time.time() - t3:.2f}s")

t4 = time.time()
print("[5] Importing dataset")
from dataset import CpGMethylationDataset
print(f"    dataset imported in {time.time() - t4:.2f}s")

t5 = time.time()
print("[6] Importing tqdm")
from tqdm import tqdm
print(f"    tqdm imported in {time.time() - t5:.2f}s")

print(f"\nALL imports finished in {time.time() - t0:.2f}s\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_chr22():
    d = np.load("per_chrom_npz/chr22.npz", allow_pickle=True)
    return d["dna"], d["histone"], d["methyl"]

def train(model, train_loader, val_loader, epochs=20, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        tr_loss = 0

        print(f"\nEpoch {epoch+1}/{epochs}")
        for batch_idx, (dna, hist, y) in enumerate(tqdm(train_loader, desc="Training")):
            dna, hist, y = dna.to(device), hist.to(device), y.to(device)
            opt.zero_grad()
            pred = model(dna, hist)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item()

            # Print every 100 steps
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}, loss={loss.item():.4f}")

        avg_train_loss = tr_loss/len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for dna, hist, y in val_loader:
                dna, hist, y = dna.to(device), hist.to(device), y.to(device)
                pred = model(dna, hist)
                val_loss += loss_fn(pred, y).item()

        avg_val_loss = val_loss/len(val_loader)
        print(f"Epoch {epoch+1}: train={avg_train_loss:.4f} | val={avg_val_loss:.4f}")

    torch.save(model.state_dict(), "chr22_model.pth")
    print("Saved model → chr22_model.pth")

if __name__ == "__main__":
    dna, hist, meth = load_chr22()
    N = len(meth)
    i1, i2 = int(0.6*N), int(0.8*N)

    train_loader = DataLoader(
        CpGMethylationDataset(dna[:i1], hist[:i1], meth[:i1]),
        batch_size=32,
        shuffle=True,
        num_workers=0,        
        pin_memory=False
    )

    val_loader   = DataLoader(
        CpGMethylationDataset(dna[i1:i2], hist[i1:i2], meth[i1:i2]),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    model = BasicMethylationNet()
    print(f"Starting training on chr22, samples={N}, train={i1}, val={i2-i1}, test={N-i2}")
    train(model, train_loader, val_loader)
