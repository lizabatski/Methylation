import torch
from torch.utils.data import Dataset

class CpGMethylationDataset(Dataset):
    def __init__(self, dna, histone, methyl):
        self.dna = torch.tensor(dna, dtype=torch.float32)
        self.histone = torch.tensor(histone, dtype=torch.float32)
        self.methyl = torch.tensor(methyl, dtype=torch.float32)

    def __len__(self):
        return len(self.methyl)

    def __getitem__(self, idx):
        return self.dna[idx], self.histone[idx], self.methyl[idx]
