import torch
import torch.nn as nn
from models.common import LSTMEncoder, RegressionHead, mean_pooling

class GatedFusionLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.text_encoder = LSTMEncoder(300, hidden_dim, num_layers, dropout)
        self.audio_encoder = LSTMEncoder(74, hidden_dim, num_layers, dropout)
        self.vision_encoder = LSTMEncoder(35, hidden_dim, num_layers, dropout)

        self.gate_layer = nn.Linear(hidden_dim * 3, hidden_dim * 3)
        self.regressor = RegressionHead(hidden_dim * 3, hidden_dim=128, dropout=dropout)

    def forward(self, text, audio, vision):
        ht = mean_pooling(self.text_encoder(text))
        ha = mean_pooling(self.audio_encoder(audio))
        hv = mean_pooling(self.vision_encoder(vision))

        h = torch.cat([ht, ha, hv], dim=-1)
        g = torch.sigmoid(self.gate_layer(h))
        fused = h * g
        out = self.regressor(fused)
        return out