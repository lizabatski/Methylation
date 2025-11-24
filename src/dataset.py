import torch
from torch.utils.data import Dataset
import numpy as np

class CpGMethylationDataset(Dataset):
    """
    Dataset for CpG methylation prediction.
    
    Args:
        dna: (N, 500, 4) - one-hot encoded DNA sequences
        histone: (N, 500, 4) - histone modification signals (H3K4me3, H3K36me2, H3K27me3, H3K9me3)
        methylation: (N,) - methylation values [0, 1]
    """
    
    def __init__(self, dna, histone, methylation):
        # Convert numpy arrays to torch tensors
        self.dna = torch.from_numpy(dna).float()
        self.histone = torch.from_numpy(histone).float()
        self.methylation = torch.from_numpy(methylation).float()
        
        # Verify all inputs have same number of samples
        assert len(self.dna) == len(self.histone) == len(self.methylation), \
            f"Mismatched lengths: DNA={len(self.dna)}, Histone={len(self.histone)}, Methyl={len(self.methylation)}"
    
    def __len__(self):
        """Return total number of samples"""
        return len(self.methylation)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            dna: (500, 4) tensor
            histone: (500, 4) tensor
            methylation: scalar tensor
        """
        return self.dna[idx], self.histone[idx], self.methylation[idx]