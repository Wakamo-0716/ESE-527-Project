import torch
import torch.nn as nn
from models.common import LSTMEncoder, RegressionHead, mean_pooling


class UnimodalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        )
        self.regressor = RegressionHead(
            input_dim=hidden_dim,
            hidden_dim=128,
            dropout=dropout
        )

    def forward(self, x):
        h = self.encoder(x)          # (B, T, H)
        pooled = mean_pooling(h)     # (B, H)
        out = self.regressor(pooled) # (B,)
        return out