import torch
import torch.nn as nn

class BasicMethylationNet(nn.Module):
    def __init__(
        self,
        dna_filters=[128, 256, 256],
        dna_kernel_sizes=[15, 11, 7],
        histone_filters=32,
        histone_kernel_size=3,
        dropout=0.3,
        fc_hidden_dims=[128, 64],
    ):
        super().__init__()
        
        # DNA pathway
        self.dna_pathway = nn.ModuleList()
        in_channels = 4
        
        for filters, kernel_size in zip(dna_filters, dna_kernel_sizes):
            self.dna_pathway.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, filters, 
                             kernel_size=kernel_size, 
                             padding=kernel_size//2),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = filters
        
        self.dna_pool = nn.AdaptiveAvgPool1d(1)
        self.dna_fc = nn.Sequential(
            nn.Linear(dna_filters[-1], fc_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Histone pathway - process each mark separately
        self.histone_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, histone_filters, 
                         kernel_size=histone_kernel_size, 
                         padding=histone_kernel_size//2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
            for _ in range(4)  # 4 histone marks
        ])
        
        self.histone_fc = nn.Sequential(
            nn.Linear(4 * histone_filters, fc_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction head
        prediction_layers = []
        input_dim = fc_hidden_dims[0] * 2
        
        for hidden_dim in fc_hidden_dims:
            prediction_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        prediction_layers.append(nn.Linear(input_dim, 1))
        self.prediction_head = nn.Sequential(*prediction_layers)
    
    def forward(self, dna, histone):
        """
        Args:
            dna: (batch, 500, 4)
            histone: (batch, 500, 4)
        Returns:
            prediction: (batch,)
        """
        dna = dna.transpose(1, 2).float()
        histone = histone.transpose(1, 2).float()
        
        # DNA pathway
        x_dna = dna
        for conv_block in self.dna_pathway:
            x_dna = conv_block(x_dna)
        x_dna = self.dna_pool(x_dna).squeeze(-1)
        x_dna = self.dna_fc(x_dna)
        
        # Histone pathway
        histone_features = []
        for i, conv in enumerate(self.histone_convs):
            mark = histone[:, i:i+1, :]
            histone_features.append(conv(mark))
        x_hist = torch.cat(histone_features, dim=1)
        x_hist = self.histone_fc(x_hist)
        
        # Combine and predict
        x = torch.cat([x_dna, x_hist], dim=1)
        out = self.prediction_head(x).squeeze(-1)
        out = torch.sigmoid(out)
        
        return out