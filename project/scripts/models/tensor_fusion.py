import torch
import torch.nn as nn
from models.common import LSTMEncoder, RegressionHead, mean_pooling

class TensorFusionLSTM(nn.Module):
    def __init__(self, hidden_dim=128, proj_dim=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.text_encoder = LSTMEncoder(300, hidden_dim, num_layers, dropout)
        self.audio_encoder = LSTMEncoder(74, hidden_dim, num_layers, dropout)
        self.vision_encoder = LSTMEncoder(35, hidden_dim, num_layers, dropout)

        self.text_proj = nn.Linear(hidden_dim, proj_dim)
        self.audio_proj = nn.Linear(hidden_dim, proj_dim)
        self.vision_proj = nn.Linear(hidden_dim, proj_dim)

        fusion_dim = (proj_dim + 1) * (proj_dim + 1) * (proj_dim + 1)
        self.regressor = RegressionHead(fusion_dim, hidden_dim=128, dropout=dropout)

    def forward(self, text, audio, vision):
        ht = self.text_proj(mean_pooling(self.text_encoder(text)))
        ha = self.audio_proj(mean_pooling(self.audio_encoder(audio)))
        hv = self.vision_proj(mean_pooling(self.vision_encoder(vision)))

        ones = torch.ones(ht.size(0), 1, device=ht.device)
        ht_ = torch.cat([ones, ht], dim=1)
        ha_ = torch.cat([ones, ha], dim=1)
        hv_ = torch.cat([ones, hv], dim=1)

        fusion_tensor = torch.einsum('bi,bj,bk->bijk', ht_, ha_, hv_)
        fused = fusion_tensor.reshape(ht.size(0), -1)

        return self.regressor(fused)