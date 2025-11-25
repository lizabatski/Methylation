import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim
import pickle

data = np.load('chr19.npz', allow_pickle=True)

class MethDataset(Dataset):
    def __init__(self, sequence, histone, methylation, coords, apply_log10=False):
        self.sequence = sequence
        self.histone = histone
        self.methylation = methylation
        self.transform = apply_log10
        self.coords = coords
        self.histone_names = ['H3K4me3', 'H3K36me2', 'H3K27me3', 'H3K9me3']

    def __len__(self):
        return self.methylation.shape[0]

    def __getitem__(self, idx):
        
        sequence = torch.from_numpy(self.sequence[idx])
        histone = self.histone.astype(np.float32)

        H3K4me3 = torch.from_numpy(histone[:, :, 0][idx].astype(np.float32)) if not self.transform else torch.from_numpy(np.log10(histone[:, :, 0]+1e-4)[idx])
        H3K36me2 = torch.from_numpy(histone[:, :, 1][idx].astype(np.float32)) if not self.transform else torch.from_numpy(np.log10(histone[:, :, 1]+1e-4)[idx])
        H3K27me3 = torch.from_numpy(histone[:, :, 2][idx].astype(np.float32)) if not self.transform else torch.from_numpy(np.log10(histone[:, :, 2]+1e-4)[idx])
        H3K9me3 = torch.from_numpy(histone[:, :, 3][idx].astype(np.float32)) if not self.transform else torch.from_numpy(np.log10(histone[:, :, 3]+1e-4)[idx])

        methylation = self.methylation[idx]
        coordinates = self.coords[idx]

        return sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, methylation, coordinates

size = data['dna'].shape[0]
split_index = int(0.8 * size) ### 80% of the data will be for training

# I'm applying log10 in both cases
train_dataset = MethDataset(sequence = data['dna'][:split_index],
                           histone = data['histone'][:split_index], 
                           methylation = data['methyl'][:split_index],
                           coords = data['coords'][:split_index],
                           apply_log10=True)


class Model(nn.Module):
    def __init__(self, DNA_kernel_sizes, DNA_strides, DNA_conv_channels):
        super().__init__()
        # Module parameters
        self.DNA_layer1_kernel_size, self.DNA_layer2_kernel_size, self.DNA_layer3_kernel_size, self.DNA_layer4_kernel_size = DNA_kernel_sizes
        self.DNA_conv_channels = DNA_conv_channels
        self.DNA_layer1_stride, self.DNA_layer2_stride, self.DNA_layer3_stride, self.DNA_layer4_stride = DNA_strides

        
        ############## Modules and architecture
        self.dna_module = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=DNA_conv_channels, kernel_size=(self.DNA_layer1_kernel_size), 
                        stride=self.DNA_layer1_stride, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=DNA_conv_channels, out_channels=1, kernel_size=(self.DNA_layer3_kernel_size), 
                        stride=self.DNA_layer3_stride, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(self.DNA_layer4_kernel_size), 
                        stride=self.DNA_layer4_stride, padding=0)
        )

        ### 
        self.H3K4me3_module = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=DNA_conv_channels, kernel_size=(self.DNA_layer1_kernel_size), 
                        stride=self.DNA_layer1_stride, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=DNA_conv_channels, out_channels=1, kernel_size=(self.DNA_layer3_kernel_size), 
                        stride=self.DNA_layer3_stride, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(self.DNA_layer4_kernel_size), 
                        stride=self.DNA_layer4_stride, padding=0)
        )
        self.H3K36me2_module = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=DNA_conv_channels, kernel_size=(self.DNA_layer1_kernel_size), 
                        stride=self.DNA_layer1_stride, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=DNA_conv_channels, out_channels=1, kernel_size=(self.DNA_layer3_kernel_size), 
                        stride=self.DNA_layer3_stride, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(self.DNA_layer4_kernel_size), 
                        stride=self.DNA_layer4_stride, padding=0)
        )
        self.H3K27me3_module = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=DNA_conv_channels, kernel_size=(self.DNA_layer1_kernel_size), 
                        stride=self.DNA_layer1_stride, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=DNA_conv_channels, out_channels=1, kernel_size=(self.DNA_layer3_kernel_size), 
                        stride=self.DNA_layer3_stride, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(self.DNA_layer4_kernel_size), 
                        stride=self.DNA_layer4_stride, padding=0)
        )
        self.H3K9me3_module = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=DNA_conv_channels, kernel_size=(self.DNA_layer1_kernel_size), 
                        stride=self.DNA_layer1_stride, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=DNA_conv_channels, out_channels=1, kernel_size=(self.DNA_layer3_kernel_size), 
                        stride=self.DNA_layer3_stride, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(self.DNA_layer4_kernel_size), 
                        stride=self.DNA_layer4_stride, padding=0)
        )
        
        #### Cross-Attention
        self.attn = nn.MultiheadAttention(embed_dim=25, num_heads=5, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(125, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Softplus()
        )

    def forward(self, sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3):
        sequence = sequence.to(torch.float32).permute(0, 2, 1) ### Changed to (B,C=4,L=500) to use Conv1D
        dna_module_output = self.dna_module(sequence)

        H3K4me3_module_output = self.H3K4me3_module(H3K4me3.unsqueeze(1))
        H3K36me2_module_output = self.H3K36me2_module(H3K36me2.unsqueeze(1))
        H3K27me3_module_output = self.H3K27me3_module(H3K27me3.unsqueeze(1))
        H3K9me3_module_output = self.H3K9me3_module(H3K9me3.unsqueeze(1))
        
        stack = torch.cat([dna_module_output, H3K4me3_module_output, H3K36me2_module_output, H3K27me3_module_output, H3K9me3_module_output], dim=1)#.permute(1,0,2) # Not sure if this is ok

        ### Attention
        attention_output, attention_weights = self.attn(stack, stack, stack)
        attention_reshaped = attention_output.reshape(attention_output.size(0), -1)
        ###

        methylation_prediction = self.fc(attention_reshaped)

        return methylation_prediction


    def training_loop(self, loss_fn, train_dataset, batch_size=10, epochs=100, learning_rate=1e-3, optimizer=torch.optim.SGD):

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optimizer(self.parameters(), lr=learning_rate)
        loss_fn = loss_fn()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model will be trained in {device}")
        self.to(device)
        
        self.train()
        loss_dict = {}
        for e in range(epochs):
            loss_accum = 0
            for i, (sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, methylation, coordinates) in enumerate(train_dataloader):
                
                sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3, methylation = sequence.to(device), H3K4me3.to(device), H3K36me2.to(device), H3K27me3.to(device), H3K9me3.to(device), methylation.to(device)
                prediction = self.forward(sequence, H3K4me3, H3K36me2, H3K27me3, H3K9me3)

                loss = loss_fn(prediction, methylation.unsqueeze(-1).float())
                
                loss_accum += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("---")

            if (e+1) % 2 == 0:
                # print(f"Iter: {e+1}, Loss: {loss_accum}")
                loss_dict[e+1] = loss_accum
        
        with open("loss_dict.pkl", "wb") as file:
            pickle.dump(loss_dict, file)
    
    def eval_loop(args, kwargs):
        pass

model = Model(DNA_kernel_sizes=(10,0,10,5), DNA_strides=(2,5,3,3), DNA_conv_channels = 2)

model.training_loop(loss_fn=nn.MSELoss, train_dataset=train_dataset, batch_size=10, epochs=150, learning_rate=1e-3, optimizer=torch.optim.Adam)