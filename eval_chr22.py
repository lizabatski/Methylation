import numpy as np, torch
from torch.utils.data import DataLoader
from model import BasicMethylationNet
from dataset import CpGMethylationDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_chr22():
    d = np.load("per_chrom_npz/chr22.npz", allow_pickle=True)
    return d["dna"], d["histone"], d["methyl"], d["coords"]

dna, hist, meth, coords = load_chr22()
N = len(meth)
test_idx = int(0.8 * N)

test_ds = CpGMethylationDataset(dna[test_idx:], hist[test_idx:], meth[test_idx:])
test_loader = DataLoader(test_ds, batch_size=32)

model = BasicMethylationNet()
model.load_state_dict(torch.load("chr22_model.pth"))
model = model.to(device).eval()

preds = []
with torch.no_grad():
    for dna_b, hist_b, _ in test_loader:
        dna_b, hist_b = dna_b.to(device), hist_b.to(device)
        p = model(dna_b, hist_b).cpu().numpy()
        preds.extend(p)

np.save("chr22_preds.npy", np.array(preds))
np.save("chr22_coords_test.npy", coords[test_idx:])
print("saved chr22_preds.npy & chr22_coords_test.npy")
