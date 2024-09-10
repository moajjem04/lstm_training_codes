'''
Create a two layer LSTM model
'''

import torch
import torch.nn as nn
import torch.autograd as autograd

class LSTMClassifier(nn.Module):
    def __init__(self, length, in_ch, hidden_size=64, num_layers=2, output_size=1, dropout=0.3, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_ch, hidden_size, num_layers, batch_first=True, bidirectional= bidirectional, dropout=dropout) 
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(in_ch)

    def forward(self, x1, x2):
        # concat x1 and x2 along axis =1
        x = torch.cat((x1, x2), axis=1)
        x = self.bn(x)
        x = x.permute(0, 2, 1) # from (N, C, L) to (N, L, C)
        # set hidden state to zero
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        # forward pass
        # out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(x)
        #print(out[:, -1, :].shape)
        out = self.fc(out[:, -1, :])
        #out = self.relu(out)
        out = self.dropout(out)
        out = nn.Softmax(dim=1)(out)
        return out

class MLPClassifier(nn.Module):
    def __init__(self, length, in_ch, hidden_size=64, num_layers=2, output_size=1, dropout=0.3, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # layers
        self.flat1 = nn.Flatten()
        self.bn1 = nn.BatchNorm1d(length)
        self.fc1 = nn.ModuleList([self._block(length, length) for _ in range(num_layers)])
        self.fc1 = nn.Sequential(*self.fc1)

        self.flat2 = nn.Flatten()
        self.bn2 = nn.BatchNorm1d(length)
        self.fc2 = nn.ModuleList([self._block(length, length) for _ in range(num_layers)])
        self.fc2 = nn.Sequential(*self.fc2)

        self.flatten = nn.Flatten()
        self.fc3 = self._block(2 * length, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, output_size)

        self.dropout_layer = nn.Dropout(dropout)    
            
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),            
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
    def forward(self, x1, x2):
        # for x1
        x1 = self.flat1(x1)
        x1 = self.bn1(x1)
        x1 = self.fc1(x1)
        # for x2
        x2 = self.flat2(x2)
        x2 = self.bn2(x2)
        x2 = self.fc2(x2)

        # concat x1 and x2 along axis =1
        x = torch.cat((x1, x2), axis=1)

        x = self.flatten(x)
        #
        x = self.fc3(x)
        out = self.fc4(x)
        #out = self.relu(out)
        out = self.dropout_layer(out)
        out = nn.Softmax(dim=1)(out)
        return out


class CNNClassifier(nn.Module):
    def __init__(self, length, in_ch, hidden_size=64, num_layers=2, output_size=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # ch = 2
        cnn = [self._cnn_block(in_ch * 2**(i), in_ch * 2**(i+1), 3, padding= 'same', dropout=dropout) for i in range(num_layers)]
        #cnn = nn.ModuleList(cnn) # convert to nn.ModuleList
        self.cnn = nn.Sequential(*cnn) # convert to a sequential graph
        #self.cnn = self._cnn_block(in_ch, in_ch *2, 3, padding= 'same')
        # num_layers = 1
        self.flatten = nn.Flatten()
        self.final_dimension =  in_ch * length # (in_ch * 2** num_layers) * (length//2**num_layers) = in_ch * length
        self.fc1 = nn.Linear(self.final_dimension, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def _cnn_block(self, in_channels, out_channels, kernel_size, padding, dropout=0.3):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2):
        # concat x1 and x2 along axis =1
        x = torch.cat((x1, x2), axis=1)
        #print(x.shape)
        out = self.cnn(x)
        out = self.flatten(out)
        #print(out.shape)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = nn.Softmax(dim=1)(out)
        #print(out.shape)
        return out