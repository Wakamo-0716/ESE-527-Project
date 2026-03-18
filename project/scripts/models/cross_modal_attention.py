import torch
import torch.nn as nn
from models.common import LSTMEncoder, RegressionHead, mean_pooling

class CrossModalAttentionLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, dropout=0.2, num_heads=4):
        super().__init__()
        self.text_encoder = LSTMEncoder(300, hidden_dim, num_layers, dropout)
        self.audio_encoder = LSTMEncoder(74, hidden_dim, num_layers, dropout)
        self.vision_encoder = LSTMEncoder(35, hidden_dim, num_layers, dropout)

        self.attn_ta = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.attn_tv = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.regressor = RegressionHead(hidden_dim * 3, hidden_dim=128, dropout=dropout)

    def forward(self, text, audio, vision):
        ht = self.text_encoder(text)    # (B,T,H)
        ha = self.audio_encoder(audio)
        hv = self.vision_encoder(vision)

        attn_ta, _ = self.attn_ta(query=ht, key=ha, value=ha)
        attn_tv, _ = self.attn_tv(query=ht, key=hv, value=hv)

        pt = mean_pooling(ht)
        pta = mean_pooling(attn_ta)
        ptv = mean_pooling(attn_tv)

        fused = torch.cat([pt, pta, ptv], dim=-1)
        return self.regressor(fused)