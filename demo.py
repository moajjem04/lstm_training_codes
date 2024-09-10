import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Example EEG data and labels (replace with your actual data)
eeg_data = np.random.randn(1000, 128, 64)  # 1000 samples, 128 time steps, 64 features
labels = np.random.randint(0, 2, size=(1000,))  # Binary labels

# Convert data to PyTorch tensors
eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dataset and dataloader
dataset = EEGDataset(eeg_data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  # Get the last time step output
        out = self.fc(out)
        return out

# Hyperparameters
input_size = 64  # Number of features
hidden_size = 128
num_layers = 2
num_classes = 2

model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    for signals, labels in dataloader:
        outputs = model(signals)
        print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
