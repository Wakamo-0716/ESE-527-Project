# import torch
# import torch.nn as nn
# from models.common import LSTMEncoder, RegressionHead, mean_pooling
#
# class TensorFusionLSTM(nn.Module):
#     def __init__(self, hidden_dim=128, proj_dim=32, num_layers=1, dropout=0.2):
#         super().__init__()
#         self.text_encoder = LSTMEncoder(300, hidden_dim, num_layers, dropout)
#         self.audio_encoder = LSTMEncoder(74, hidden_dim, num_layers, dropout)
#         self.vision_encoder = LSTMEncoder(35, hidden_dim, num_layers, dropout)
#
#         self.text_proj = nn.Linear(hidden_dim, proj_dim)
#         self.audio_proj = nn.Linear(hidden_dim, proj_dim)
#         self.vision_proj = nn.Linear(hidden_dim, proj_dim)
#
#         fusion_dim = (proj_dim + 1) * (proj_dim + 1) * (proj_dim + 1)
#         self.regressor = RegressionHead(fusion_dim, hidden_dim=128, dropout=dropout)
#
#     def forward(self, text, audio, vision):
#         ht = self.text_proj(mean_pooling(self.text_encoder(text)))
#         ha = self.audio_proj(mean_pooling(self.audio_encoder(audio)))
#         hv = self.vision_proj(mean_pooling(self.vision_encoder(vision)))
#
#         ones = torch.ones(ht.size(0), 1, device=ht.device)
#         ht_ = torch.cat([ones, ht], dim=1)
#         ha_ = torch.cat([ones, ha], dim=1)
#         hv_ = torch.cat([ones, hv], dim=1)
#
#         fusion_tensor = torch.einsum('bi,bj,bk->bijk', ht_, ha_, hv_)
#         fused = fusion_tensor.reshape(ht.size(0), -1)
#
#         return self.regressor(fused)

import torch
import torch.nn as nn
from models.common import (
    LSTMEncoder,
    RegressionHead,
    mean_pooling,
    ModalityProjection,
)


class TensorFusionLSTM(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        proj_dim=64,
        tensor_proj_dim=32,
        num_layers=1,
        dropout=0.2
    ):
        super().__init__()

        # Input projection to shared dimension
        self.text_proj = ModalityProjection(300, proj_dim, dropout)
        self.audio_proj = ModalityProjection(74, proj_dim, dropout)
        self.vision_proj = ModalityProjection(35, proj_dim, dropout)

        # Sequence encoders
        self.text_encoder = LSTMEncoder(proj_dim, hidden_dim, num_layers, dropout)
        self.audio_encoder = LSTMEncoder(proj_dim, hidden_dim, num_layers, dropout)
        self.vision_encoder = LSTMEncoder(proj_dim, hidden_dim, num_layers, dropout)

        # Project pooled hidden states to a smaller dimension before tensor fusion
        self.text_post = nn.Linear(hidden_dim, tensor_proj_dim)
        self.audio_post = nn.Linear(hidden_dim, tensor_proj_dim)
        self.vision_post = nn.Linear(hidden_dim, tensor_proj_dim)

        fusion_dim = (tensor_proj_dim + 1) * (tensor_proj_dim + 1) * (tensor_proj_dim + 1)

        self.regressor = RegressionHead(
            input_dim=fusion_dim,
            hidden_dim=128,
            dropout=dropout
        )

    def forward(self, text, audio, vision):
        text = self.text_proj(text)
        audio = self.audio_proj(audio)
        vision = self.vision_proj(vision)

        ht = mean_pooling(self.text_encoder(text))
        ha = mean_pooling(self.audio_encoder(audio))
        hv = mean_pooling(self.vision_encoder(vision))

        ht = self.text_post(ht)
        ha = self.audio_post(ha)
        hv = self.vision_post(hv)

        ones = torch.ones(ht.size(0), 1, device=ht.device)

        ht_ = torch.cat([ones, ht], dim=1)
        ha_ = torch.cat([ones, ha], dim=1)
        hv_ = torch.cat([ones, hv], dim=1)

        fusion_tensor = torch.einsum("bi,bj,bk->bijk", ht_, ha_, hv_)
        fused = fusion_tensor.reshape(ht.size(0), -1)

        out = self.regressor(fused)
        return out