# import torch
# import torch.nn as nn
# from models.common import LSTMEncoder, RegressionHead, mean_pooling
#
# class EarlyFusionLSTM(nn.Module):
#     def __init__(self, hidden_dim=128, num_layers=1, dropout=0.2):
#         super().__init__()
#         self.encoder = LSTMEncoder(
#             input_dim=300 + 74 + 35,
#             hidden_dim=hidden_dim,
#             num_layers=num_layers,
#             dropout=dropout,
#             bidirectional=False
#         )
#         self.regressor = RegressionHead(hidden_dim, hidden_dim=128, dropout=dropout)
#
#     def forward(self, text, audio, vision):
#         x = torch.cat([text, audio, vision], dim=-1)  # (B, T, 409)
#         h = self.encoder(x)                           # (B, T, H)
#         pooled = mean_pooling(h)                      # (B, H)
#         out = self.regressor(pooled)                  # (B,)
#         return out

import torch
import torch.nn as nn
from models.common import (
    LSTMEncoder,
    RegressionHead,
    mean_pooling,
    ModalityProjection,
)


class EarlyFusionLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, dropout=0.2, proj_dim=64):
        super().__init__()

        # Project each modality to the same dimension first
        self.text_proj = ModalityProjection(300, proj_dim, dropout)
        self.audio_proj = ModalityProjection(74, proj_dim, dropout)
        self.vision_proj = ModalityProjection(35, proj_dim, dropout)

        # Early fusion after projection
        self.encoder = LSTMEncoder(
            input_dim=proj_dim * 3,
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

    def forward(self, text, audio, vision):
        text = self.text_proj(text)        # (B, T, proj_dim)
        audio = self.audio_proj(audio)     # (B, T, proj_dim)
        vision = self.vision_proj(vision)  # (B, T, proj_dim)

        x = torch.cat([text, audio, vision], dim=-1)  # (B, T, 3*proj_dim)
        h = self.encoder(x)                           # (B, T, H)
        pooled = mean_pooling(h)                      # (B, H)
        out = self.regressor(pooled)                  # (B,)
        return out