import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicMethylationNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.dna_conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.dna_fc = nn.Linear(64 * 125, 128)

        self.hist_conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.hist_fc = nn.Linear(32 * 5, 128)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, dna, histone):
        x_dna = self.dna_conv(dna)
        x_dna = x_dna.reshape(x_dna.size(0), -1)
        x_dna = F.relu(self.dna_fc(x_dna))

        x_hist = self.hist_conv(histone)
        x_hist = x_hist.reshape(x_hist.size(0), -1)
        x_hist = F.relu(self.hist_fc(x_hist))

        x = torch.cat([x_dna, x_hist], dim=1)
        x = self.fc(x)
        return x.squeeze()
