import torch
import torch.nn as nn
from params import params


class OSRENet(nn.Module):
    def __init__(self):
        super(OSRENet, self).__init__()

        self.lstm = nn.LSTM(input_size=params.feature_dim, hidden_size=64, num_layers=2, batch_first=True)


    def forward(self, x):
        x = self.lstm
        return x
